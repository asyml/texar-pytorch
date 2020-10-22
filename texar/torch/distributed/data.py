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

from typing import (Optional, Sequence)
from adaptdl.torch.data import AdaptiveDataLoader

from texar.torch.data.data.sampler import BatchingStrategy
from texar.torch.data.data.data_base import DatasetBase
from texar.torch.data.data.data_iterators import (DatasetsType, DataIterator)


class AdaptiveDataIterator(DataIterator):
    def __init__(self, datasets: DatasetsType,
                 batching_strategy: Optional[BatchingStrategy] = None,
                 pin_memory: Optional[bool] = None):
        self._default_dataset_name = 'data'
        if isinstance(datasets, DatasetBase):
            datasets = {self._default_dataset_name: datasets}
        elif isinstance(datasets, Sequence):
            if any(not isinstance(d, DatasetBase) for d in datasets):
                raise ValueError("`datasets` must be an non-empty list of "
                                 "`texar.torch.data.DatasetBase` instances.")
            num_datasets = len(datasets)
            datasets = {d.name: d for d in datasets}
            if len(datasets) < num_datasets:
                raise ValueError("Names of datasets must be unique.")

        self._datasets = {
            name: AdaptiveDataLoader(dataset,
                                     batch_size=dataset.batch_size,
                                     shuffle=dataset.hparams.shuffle,
                                     collate_fn=dataset.collate,
                                     pin_memory=pin_memory)
            for name, dataset in datasets.items()}

        if len(self._datasets) <= 0:
            raise ValueError("`datasets` must not be empty.")

        self._current_dataset_name: Optional[str] = None
