# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base data class that is inherited by all data classes.
A data defines data reading, parsing, batching, and other
preprocessing operations.
"""
from abc import ABC
from typing import Callable, List, TypeVar, Generic

from torch.utils.data import Dataset

from texar.data.data.dataset_utils import Batch
from texar.hyperparams import HParams

__all__ = [
    "DataBase"
]

Example = TypeVar('Example')  # type of a single data example


class DataBase(Dataset, Generic[Example], ABC):
    r"""Base class inherited by all data classes.
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                "num_epochs": 1,
                "batch_size": 64,
                "allow_smaller_final_batch": True,
                "shuffle": True,
                "shuffle_buffer_size": None,
                "shard_and_shuffle": False,
                "num_parallel_calls": 1,
                "prefetch_buffer_size": 0,
                "max_dataset_size": -1,
                "seed": None,
                "name": "data",
            }

        Here:

            "num_epochs" : int
                Number of times the dataset should be repeated. An
                :tf_main:`OutOfRangeError <errors/OutOfRangeError>` signal will
                be raised after the whole repeated dataset has been iterated
                through.

                E.g., For training data, set it to 1 (default) so that you
                will get the signal after each epoch of training. Set to -1
                to repeat the dataset indefinitely.

            "batch_size" : int
                Batch size, i.e., the number of consecutive elements of the
                dataset to combine in a single batch.

            "allow_smaller_final_batch" : bool
               Whether to allow the final batch to be smaller if there are
               insufficient elements left. If `False`, the final batch is
               discarded if it is smaller than batch size. Note that,
               if `True`, `output_shapes` of the resulting dataset
               will have a a **static** batch_size dimension equal to
               "batch_size".

            "shuffle" : bool
                Whether to randomly shuffle the elements of the dataset.

            "shuffle_buffer_size" : int
                The buffer size for data shuffling. The larger, the better
                the resulting data is mixed.

                If `None` (default), buffer size is set to the size of the
                whole dataset (i.e., make the shuffling the maximally
                effective).

            "shard_and_shuffle" : bool
                Whether to first shard the dataset and then shuffle each
                block respectively. Useful when the whole data is too large to
                be loaded efficiently into the memory.

                If `True`, :attr:`shuffle_buffer_size` must be specified to
                determine the size of each shard.

            "num_parallel_calls" : int
                Number of elements from the datasets to process in parallel.

            "prefetch_buffer_size" : int
                The maximum number of elements that will be buffered when
                prefetching.

            max_dataset_size : int
                Maximum number of instances to include in
                the dataset. If set to `-1` or greater than the size of
                dataset, all instances will be included. This constraint is
                imposed after data shuffling and filtering.

            seed : int, optional
                The random seed for shuffle.

                Note that if a seed is set, the shuffle order will be exact
                the same every time when going through the (repeated) dataset.

                For example, consider a dataset with elements [1, 2, 3], with
                "num_epochs"`=2` and some fixed seed, the resulting sequence
                can be: 2 1 3, 1 3 2 | 2 1 3, 1 3 2, ... That is, the orders are
                different **within** every `num_epochs`, but are the same
                **across** the `num_epochs`.

            name : str
                Name of the data.
        """
        return {
            "name": "data",
            "num_epochs": 1,
            "batch_size": 64,
            "allow_smaller_final_batch": True,
            "shuffle": True,
            "shuffle_buffer_size": None,
            "shard_and_shuffle": False,
            "num_parallel_calls": 1,
            "prefetch_buffer_size": 0,
            "max_dataset_size": -1,
            "seed": None
        }

    @property
    def num_epochs(self):
        r"""Number of epochs.
        """
        return self._hparams.num_epochs

    @property
    def batch_size(self):
        r"""The batch size.
        """
        return self._hparams.batch_size

    @property
    def hparams(self):
        r"""A :class:`~texar.HParams` instance of the
        data hyperparameters.
        """
        return self._hparams

    @property
    def name(self):
        r"""Name of the module.
        """
        return self._hparams.name

    @property
    def collate_fn(self) -> Callable[[List[Example]], Batch]:
        r"""Create a `collate_fn` for :class:`~torch.utils.data.DataLoader` that
        is used to combine examples into batches. This function takes a list of
        examples, and return an instance of :class:`~texar.data.data.Batch`.
        Implementation should make sure that the returned callable does not
        depend on `self`, and should be multi-processing-safe.

        Returns:
            A callable `collate_fn`.
        """
        raise NotImplementedError
