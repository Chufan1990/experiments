import numpy as np
import pandas as pd
import os
import logging
import torch

from math import ceil
from collections import defaultdict
from typing import List, Union

logger = logging.getLogger(__name__)


def one_hot_encode(index, num_max, leave_one_out=True):
    res = np.array([int(i == index) for i in range(num_max)])
    if leave_one_out:
        return res[1:]
    return res


class GaussianPreprocessor(object):
    def __init__(
        self,
        data_paths,
        masks=None,
        normalized_features: Union[List[str], None] = None,
        onehot_features: Union[List[str], None] = None,
        bounds=None,
        filtermasks=None,
        filtervalues=None,
    ):
        self.data_paths = data_paths
        self.masks = masks
        self.data_df = None
        self.read_csv()
        self.bounds = self.generate_bounds(bounds)
        self.normalized_features = normalized_features
        self.onehot_features = onehot_features
        self.data_df = None
        self.filtermasks = filtermasks
        self.filtervalues = filtervalues

    def read_csv(self):
        for p in self.data_paths:
            for fn in os.listdir(p):
                full_fn = os.path.join(p, fn)
                if os.path.isfile(full_fn) and full_fn.endswith(".csv"):
                    self.data_df = pd.concat(
                        [self.data_df, pd.read_csv(full_fn)], axis=0, ignore_index=True
                    )

        self.data_df.reset_index()

    def generate_bounds(self, bounds):
        if bounds is not None:
            return bounds

        if self.data_df is None:
            raise Exception("Not data frame specified yet")

        return [self.data_df.min(), self.data_df.max()]

    def features(self, x: pd.DataFrame):
        selected_features = [
            feature for feature in x.columns if feature not in self.masks
        ]
        return x[selected_features]

    def normalize(self, x: pd.DataFrame):
        if self.normalized_features is None:
            return x

        columns = self.normalized_features

        lb = self.bounds[0]
        ub = self.bounds[1]

        x[columns] = x[columns].apply(
            lambda x: (x[columns] - lb[columns]).div(ub[columns] - lb[columns]) * 2.0
            - 1.0,
            axis=1,
        )

        return x

    def onehot(self, x: pd.DataFrame):
        if self.onehot_features is None:
            return x

        oh_df = pd.DataFrame()

        for col in self.onehot_features:

            num_max = int(self.bounds[1][col])

            old_columns = oh_df.columns.tolist()

            oh_df = pd.concat(
                [
                    oh_df,
                    x[col].apply(
                        lambda r: pd.Series(one_hot_encode(int(r), num_max + 1, False))
                    ),
                ],
                axis=1,
            )
            oh_df.set_axis(
                old_columns + [col + "_" + str(i) for i in range(num_max + 1)],
                axis="columns",
            )
        x.pop(col)
        x = pd.concat([x, oh_df], axis=1)
        return x

    def prepare_df(self, x: pd.DataFrame):
        x = self.features(x)
        x = self.normalize(x)
        x = self.onehot(x)
        print(x.columns)
        return x

    def preprocess_df_sequence(self, seq):
        return torch.tensor([seq.to_numpy()])

    def filter(
        self,
        seq: pd.DataFrame,
    ):
        masks = self.filtermasks
        maskout = self.filtervalues

        if masks is None or maskout is None:
            return True

        for _, row in seq.iterrows():
            for mask, out in zip(masks, maskout):
                if row[mask] in out:
                    return False
        return True


class BasicDataset(object):
    def __init__(
        self,
        data_paths,
        batch_size,
        preprocesser,
        augmentations=None,
        include_ids=None,
        shuffle=True,
        prepare_on_load=True,
    ):
        self.debug = False
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.preprocessor = preprocesser
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.prepare_on_load = prepare_on_load
        self.data_d = {}
        self.load_data(include_ids)

    @property
    def size(self) -> int:
        return len(self.data_d.keys())

    @property
    def batches_per_epoch(self) -> int:
        return ceil(self.size / self.batch_size)

    def load_data(self, include_ids=None):
        if include_ids is not None:
            include_ids = set(include_ids)

        for p in self.data_paths:
            for fn in os.listdir(p):
                if include_ids is not None and fn not in include_ids:
                    continue
                full_fn = os.path.join(p, fn)

                if full_fn in self.data_d:
                    logger.warning(f"Duplicate file name {fn}")
                self.data_d[full_fn] = (
                    self.preprocessor.prepare_df(pd.read_csv(full_fn))
                    if self.prepare_on_load
                    else pd.read_csv(full_fn)
                )

    def batches(self):
        raise NotImplementedError()


class SequenceDataset(BasicDataset):
    def __init__(
        self,
        data_paths,
        batch_size,
        preprocessor,
        augmentations=None,
        include_ids=None,
        shuffle=True,
        prepare_on_load=True,
        num_actions=0,
        seq_length=10,
        leave_one_out_encoding=True,
    ):
        super().__init__(
            data_paths,
            batch_size,
            preprocessor,
            augmentations,
            include_ids,
            shuffle,
            prepare_on_load,
        )
        if seq_length < 2:
            raise ValueError("Need sequence length of at least 2")
        self.with_actions = num_actions > 0
        self.num_actions = num_actions
        self.seq_length = seq_length
        self.leave_one_out = leave_one_out_encoding
        self.sequences = []
        self.generate_sequences()

    def generate_sequences(self):
        self.sequences = defaultdict(list)

        for seq_id, df in self.data_d.items():
            print(seq_id, df)

            if df.shape[0] < self.seq_length:
                logger.warning(
                    f"Sequence {seq_id} is too short for sequence length {self.seq_length}"
                )
                continue

            index = df.index.tolist()
            lst_idx = 0
            while lst_idx + self.seq_length <= df.shape[0]:
                seq = index[lst_idx : lst_idx + self.seq_length]
                if self.preprocessor.filter(
                    df.iloc[index[lst_idx : lst_idx + self.seq_length]]
                ):
                    self.sequences[seq_id].append(seq)
                lst_idx += 1

        logger.info(
            f"Generated {len(self.sequences)} sequences of length {self.seq_length}"
        )

    def load_data(self, include_ids=None):
        self.data_d = defaultdict(pd.DataFrame)
        if isinstance(include_ids, dict) and {
            type(v) for v in include_ids.values()
        } == {list}:
            for p in self.data_paths:
                print(self.data_paths)
                for seq_id, lst in include_ids.items():
                    print(seq_id, lst)
                    for obj in lst:
                        fn = obj["fn"]
                        full_fn = os.path.join(p, fn)
                        if not os.path.exists(full_fn):
                            continue

                        df = (
                            self.preprocessor.prepare_df(pd.read_csv(full_fn))
                            if self.prepare_on_load
                            else pd.read_csv(full_fn)
                        )
                        self.data_d[seq_id] = df
        else:
            raise NotImplementedError()

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
        augmentations=None,
        include_ids=None,
        shuffle=True,
        prepare_on_load=True,
        num_actions=0,
        seq_length=10,
        leave_one_out_encoding=True,
    ):
        super().__init__(
            data_paths,
            batch_size,
            preprocessor,
            augmentations,
            include_ids,
            shuffle,
            prepare_on_load,
            num_actions,
            seq_length,
            leave_one_out_encoding,
        )

    def batches(self):
        data = self.sequences

        if self.shuffle:
            for key, lst in self.sequences.items():
                data[key] = [lst[i] for i in np.random.permutation(len(lst))]

        for key, lst in data.items():

            curr_idx = 0

            while curr_idx < len(lst):
                x_batch_list = []

                while curr_idx < len(lst) and len(x_batch_list) < 100:
                    seq_df = self.data_d[key].iloc[lst[curr_idx]]

                    x_tensor = self.preprocessor.preprocess_df_sequence(seq_df)

                    x_batch_list.append(x_tensor)

                    curr_idx += 1

                x_batch = torch.cat(x_batch_list, dim=0)

                yield x_batch

    def sample(self):

        data = self.sequences

        for key, lst in self.sequences.items():
            data[key] = [lst[i] for i in np.random.permutation(len(lst))]

        keys = list(data.keys())
        key = keys[np.random.randint(len(data.keys()))]

        lst = data[key]

        seq_df = self.data_d[key].iloc[lst[0]]

        return self.preprocessor.preprocess_df_sequence(seq_df)


class SequencePredictionDataset(SequenceDataset):
    def __init__(
        self,
        data_paths,
        batch_size,
        preprocessor,
        augmentations=None,
        include_ids=None,
        shuffle=True,
        prepare_on_load=True,
        num_actions=0,
        seq_length=10,
        forecast_length=5,
        leave_one_out_encoding=True,
    ):
        super().__init__(
            data_paths,
            batch_size,
            preprocessor,
            augmentations,
            include_ids,
            shuffle,
            prepare_on_load,
            num_actions,
            seq_length,
            leave_one_out_encoding,
        )
        if forecast_length >= seq_length:
            raise Exception("forecasting sequence is too long")
        self.forecast_length = forecast_length

    def batches(self):
        data = self.sequences

        if self.shuffle:
            for key, lst in self.sequences.items():
                print("key {}".format(key))
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