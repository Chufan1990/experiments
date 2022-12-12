import numpy as np
import pandas as pd
import os
import logging
import torch
import concurrent
import json
import re
import tqdm

from math import ceil
from collections import defaultdict
from typing import List, Union

logger = logging.getLogger(__name__)

def flatten_dict(x: dict, prefix=""):
    y = {}
    for k, v in x.items():
        if isinstance(v, dict):
            y.update(flatten_dict(v, k + "_"))
        else:
            y[prefix + k] = v
    return y

def one_hot_encode(index, num_max, leave_one_out=True):
    res = np.array([int(i == index) for i in range(num_max)])
    if leave_one_out:
        return res[1:]
    return res

def num_of_lines(abs_path):
    num = 0
    if not os.path.exists(abs_path):
        return num
    with open(abs_path) as file:
        for line in file:
            num += 1
    return num

def read_dict(filepath):
    tmp = []
    with open(filepath) as f:
        for line in f:
            tmp.append(flatten_dict(json.loads(line)))
    return tmp


class GaussianPreprocessor(object):
    def __init__(
        self,
        data_paths,
        features: Union[List[str], None] = None,
        normalize_features: Union[List[str], None] = None,
        onehot_features: Union[List[str], None] = None,
        lb: np.array = None,
        ub: np.array = None,
    ):
        self.data_paths = data_paths
        self.features = features
        self.normalize_features = normalize_features
        self.onehot_features = onehot_features
        self.lb = lb
        self.ub = ub

    def mask(self, x: pd.DataFrame, features=None):
        if features is None:
            return x    
        return x[features]

    def normalize(self, x: pd.DataFrame, features=None, lb=None, ub=None):
        if features is None or lb is None or ub is None:
            return x

        x[features] = (x[features] - lb) / (ub - lb) * 2.0 - 1.0
        return x

    def onehot(self, x: pd.DataFrame, features=None, lb=None, ub=None):
        if features is None or lb is None or ub is None:
            return x

        oh_df = pd.DataFrame()

        for i, col in enumerate(features):

            if col not in x.columns.tolist():
                continue

            num_max = ub[i] - lb[i]

            old_columns = oh_df.columns.tolist()

            oh_df = pd.concat(
                [
                    oh_df,
                    x[col].apply(
                        lambda r: pd.Series(one_hot_encode(int(r - lb[i]), num_max + 1, False))
                    ),
                ],
                axis=1,
            )
            oh_df.set_axis(
                old_columns + [col + "_" + str(i) for i in range(num_max + 1)],
                axis="columns",
            )
        x.pop(col)
        return pd.concat([x, oh_df], axis=1)

    def prepare_df(self, x: pd.DataFrame):
        x["throttle"] -= x.pop("brake")
        x = self.normalize(x, self.normalize_features, self.lb, self.ub)
        x = self.onehot(x, self.onehot_features, self.lb, self.ub)
        return x

    def preprocess_df_sequence(self, seq_d):
        tmp = []
        for key, val in seq_d.items():
            tmp.append(self.mask(val, self.features[key]))
        df = pd.concat(tmp, axis=1)
        return torch.tensor([self.prepare_df(df).to_numpy()])

    def filter(
        self,
        seq: pd.DataFrame,
    ):
        for _, row in seq.iterrows():
            if row["drivingMode"] != "COMPLETE_AUTO_DRIVE":
                return False
        return True


class BasicDataset(object):
    def __init__(
        self,
        data_paths,
        batch_size,
        preprocessor,
        files,
        augmentations=None,
        include_dirs=None,
        shuffle=True,
        prepare_on_load=True,
        args=None
    ):
        self.debug = False
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.files = files
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.prepare_on_load = prepare_on_load
        self.data_d = defaultdict(dict)
        self.args = args
        self.load_data(include_dirs, args)

    @property
    def size(self) -> int:
        return len(self.data_d.keys())

    @property
    def batches_per_epoch(self) -> int:
        return ceil(self.size / self.batch_size)

    def load_data(self, include_dirs=None):
        if include_dirs is not None:
            include_dirs = set(include_dirs)


        for path in self.data_paths:
            for case in os.listdir(path):
                if include_dirs is not None and case not in include_dirs:
                    continue

                incompleted_files = False
                file_length = None
                tmp = defaultdict(pd.DataFrame) if self.in_memory else defaultdict(str)
                for f in self.files:
                    filepath = os.path.join(path, case, f)
                    if file_length is not None and file_length != num_of_lines(filepath):
                        incompleted_files = True
                        break
                    file_length = num_of_lines(filepath)

                    if self.args.in_memory:
                        tmp[f] = pd.DataFrame(read_dict(filepath))
                    else:
                        df = pd.DataFrame(read_dict(filepath))
                        filename_in_csv = re.sub(".txt$", ".csv", filepath)
                        df.to_csv(filename_in_csv)
                        tmp[f] = filename_in_csv

                if incompleted_files:
                    continue
                
                self.data_d[case] = tmp

    def batches(self):
        raise NotImplementedError()


class SequenceDataset(BasicDataset):
    def __init__(
        self,
        data_paths,
        batch_size,
        preprocessor,
        files,
        augmentations=None,
        include_dirs=None,
        shuffle=True,
        prepare_on_load=True,
        num_actions=0,
        seq_length=10,
        leave_one_out_encoding=True,
        args=None,
    ):
        super().__init__(
            data_paths,
            batch_size,
            preprocessor,
            files,
            augmentations,
            include_dirs,
            shuffle,
            prepare_on_load,
            args
        )
        if seq_length < 2:
            raise ValueError("Need sequence length of at least 2")
        self.with_actions = num_actions > 0
        self.num_actions = num_actions
        self.seq_length = seq_length
        self.leave_one_out = leave_one_out_encoding
        self.sequences = []
        self.generate_sequences(args)

    def generate_sequences(self, args):
        self.sequences = defaultdict(list)

        for seq_id, df_dict in self.data_d.items():

            df = df_dict[self.files[0]] if args.in_memory else pd.read_csv(df_dict[self.files[0]])
            num_samples = df.shape[0]
            if num_samples < self.seq_length:
                logger.warning(
                    f"Sequence {seq_id} is too short for sequence length {self.seq_length}"
                )
                continue

            index = df.index.tolist()
            lst_idx = 0
            while lst_idx + self.seq_length <= num_samples:
                seq = index[lst_idx : lst_idx + self.seq_length]
                if self.preprocessor.filter(
                    df.iloc[index[lst_idx : lst_idx + self.seq_length]]
                ):
                    self.sequences[seq_id].append(seq)
                lst_idx += 1

        logger.info(
            f"Generated {len(self.sequences)} sequences of length {self.seq_length}"
        )

    def load_data(self, include_dirs, args):

        self.data_d = defaultdict(list)

        if not args.in_memory:
            tasks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                for path in self.data_paths:
                    for case in os.listdir(path):
                        if include_dirs is not None and case not in include_dirs:
                            continue
                        tasks.append(executor.submit(self._load_data, path, case))
                for future in tqdm.tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
                    try:
                        case, dictionary = future.result()
                        if dictionary is not None:
                            self.data_d[case] = dictionary
                    except Exception as e:
                        logger.error(e)
        else:
            for path in self.data_paths:
                for case in os.listdir(path):
                    if include_dirs is not None and case not in include_dirs:
                        continue

                    incompleted_files = False
                    file_length = None
                    tmp = defaultdict(pd.DataFrame)
                    for f in self.files:
                        filepath = os.path.join(path, case, f)
                        if file_length is not None and file_length != num_of_lines(filepath):
                            incompleted_files = True
                            break
                        file_length = num_of_lines(filepath)
                        tmp[f] = pd.DataFrame(read_dict(filepath))

                    if incompleted_files:
                            continue
                        
                    self.data_d[case] = tmp

    def _load_data(self, path, case):
        incompleted_files = False
        file_length = None
        tmp = defaultdict(str)
        for f in self.files:
            filepath = os.path.join(path, case, f)
            filename_in_csv = re.sub(".txt$", ".csv", filepath)

            if file_length is not None and file_length != num_of_lines(filepath):
                incompleted_files = True
                break
            file_length = num_of_lines(filepath)

            df =  pd.DataFrame(read_dict(filepath))
            df.to_csv(filename_in_csv)

            tmp[f] = filename_in_csv

        if incompleted_files:
            return case, None
        return case, tmp
                    
    def actions_to_tensor(self, seq):
        raise NotImplementedError()

    def batches(self):
        return NotImplementedError()


class SequenceReconstructionDataset(SequenceDataset):
    def __init__(
        self,
        data_paths,
        batch_size,
        preprocessor,
        files,
        augmentations=None,
        include_dirs=None,
        shuffle=True,
        prepare_on_load=True,
        num_actions=0,
        seq_length=10,
        leave_one_out_encoding=True,
        args=None
    ):
        super().__init__(
            data_paths,
            batch_size,
            preprocessor,
            files,
            augmentations,
            include_dirs,
            shuffle,
            prepare_on_load,
            num_actions,
            seq_length,
            leave_one_out_encoding,
            args
        )

    def batches(self):
        data = self.sequences

        # if self.shuffle:
        #     for key, lst in self.sequences.items():
        #         data[key] = [lst[i] for i in np.random.permutation(len(lst))]

        keys = [list(self.sequences.keys())[i] for i in np.random.permutation(len(self.sequences.keys()))]

        for key in keys:
            lst = [data[key][i] for i in np.random.permutation(len(data[key]))]
            curr_idx = 0
            while curr_idx < len(lst):
                x_batch_list = []

                while curr_idx < len(lst) and len(x_batch_list) < 100:
                    seq_df_d = {}
                    for file in self.files:
                        seq_df_d[file] = self.data_d[key][file].iloc[lst[curr_idx]] if self.args.in_memory \
                            else pd.read_csv(self.data_d[key][file]).iloc[lst[curr_idx]]
                    x_tensor = self.preprocessor.preprocess_df_sequence(seq_df_d)

                    if x_tensor is not None:
                        x_batch_list.append(x_tensor)

                    curr_idx += 1

                x_batch = torch.cat(x_batch_list, dim=0)

                yield x_batch

    def sample(self):
        data = self.sequences

        keys = [list(self.sequences.keys())[i] for i in np.random.permutation(len(self.sequences.keys()))]
        lst = [data[keys[0]][i] for i in np.random.permutation(len(data[keys[0]]))]

        key = keys[0]
        seq_df_d = {}
        for file in self.files:
            seq_df_d[file] = self.data_d[key][file].iloc[lst[0]] if self.args.in_memory \
                else pd.read_csv(self.data_d[key][file]).iloc[lst[0]]
     
        return self.preprocessor.preprocess_df_sequence(seq_df_d)


class SequencePredictionDataset(SequenceDataset):
    def __init__(
        self,
        data_paths,
        batch_size,
        preprocessor,
        file,
        augmentations=None,
        include_dirs=None,
        shuffle=True,
        prepare_on_load=True,
        num_actions=0,
        seq_length=10,
        forecast_length=5,
        leave_one_out_encoding=True,
        args=None
    ):
        super().__init__(
            data_paths,
            batch_size,
            preprocessor,
            file,
            augmentations,
            include_dirs,
            shuffle,
            prepare_on_load,
            num_actions,
            seq_length,
            leave_one_out_encoding,
            args
        )
        if forecast_length >= seq_length:
            raise Exception("forecasting sequence is too long")
        self.forecast_length = forecast_length

    def batches(self):
        data = self.sequences

        if self.shuffle:
            for key, lst in self.sequences.items():
                # print("key {}".format(key))
                data[key] = [lst[i] for i in np.random.permutation(len(lst))]

        for key, lst in data.items():

            curr_idx = 0
            while curr_idx < len(lst):
                x_batch_list = []
                y_batch_list = []
                # a_batch_list = []

                while curr_idx < len(lst) and len(x_batch_list) < self.batch_size:
                    seq_df = self.data_d[key].iloc[lst[curr_idx]]
                    in_seq = seq_df.iloc[: -self.forecast_length]
                    out_seq = seq_df.iloc[-self.forecast_length :]

                    if self.augmentations is not None:
                        # TODO implement
                        pass

                    x_tensor = self.preprocessor.preprocess_df_sequence(in_seq)
                    y_tensor = self.preprocessor.preprocess_df_sequence(out_seq)

                    # if self.num_actions > 0:
                    #     a_batch_list.append(
                    #         self.actions_to_tensor(
                    #             [e["action"] for e in data[curr_idx][:-1]]
                    #         )
                    #     )

                    x_batch_list.append(x_tensor)
                    y_batch_list.append(y_tensor)

                    curr_idx += 1

                x_batch = torch.cat(x_batch_list, dim=0)
                # a_batch = (
                #     None if len(a_batch_list) == 0 else torch.cat(a_batch_list, dim=1)
                # )
                y_batch = torch.cat(y_batch_list, dim=0)

                yield x_batch, y_batch