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
Various data iterator utility classes.
"""

import pkg_resources

from torch import __version__ as _torch_version  # type: ignore
from torch.utils.data.dataloader import (  # type: ignore
    _BaseDataLoaderIter, _SingleProcessDataLoaderIter,
    _MultiProcessingDataLoaderIter)


_torch_version = pkg_resources.parse_version(_torch_version)


# PyTorch 1.3 change some attribute names in `_BaseDataLoaderIter`,
# `_SingleProcessDataLoaderIter`, and `_MultiProcessingDataLoaderIter`

if _torch_version >= pkg_resources.parse_version("1.3.0"):

    class TexarBaseDataLoaderIter(_BaseDataLoaderIter):

        @property
        def dataset(self):
            return self._dataset

        @property
        def dataset_kind(self):
            return self._dataset_kind

        @property
        def auto_collation(self):
            return self._auto_collation

        @property
        def drop_last(self):
            return self._drop_last

        @property
        def index_sampler(self):
            return self._index_sampler

        @property
        def num_workers(self):
            return self._num_workers

        @property
        def pin_memory(self):
            return self._pin_memory

        @property
        def timeout(self):
            return self._timeout

        @property
        def collate_fn(self):
            return self._collate_fn

        @property
        def sampler_iter(self):
            return self._sampler_iter

        @property
        def base_seed(self):
            return self._base_seed

    class TexarSingleProcessDataLoaderIter(_SingleProcessDataLoaderIter,
                                           TexarBaseDataLoaderIter):

        @property
        def dataset_fetcher(self):
            return self._dataset_fetcher

    class TexarMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter,
                                             TexarBaseDataLoaderIter):

        @property
        def worker_init_fn(self):
            return self._worker_init_fn

        @property
        def worker_queue_idx_cycle(self):
            return self._worker_queue_idx_cycle

        @property
        def worker_result_queue(self):
            return self._worker_result_queue

        @property
        def worker_pids_set(self):
            return self._worker_pids_set

        @property
        def shutdown(self):
            return self._shutdown

        @property
        def send_idx(self):
            return self._send_idx

        @send_idx.setter
        def send_idx(self, value):
            # pylint: disable=attribute-defined-outside-init
            self._send_idx = value

        @property
        def rcvd_idx(self):
            return self._rcvd_idx

        @property
        def task_info(self):
            return self._task_info

        @property
        def tasks_outstanding(self):
            return self._tasks_outstanding

        @tasks_outstanding.setter
        def tasks_outstanding(self, value):
            # pylint: disable=attribute-defined-outside-init
            self._tasks_outstanding = value

        @property
        def workers_done_event(self):
            return self._workers_done_event

        @property
        def index_queues(self):
            return self._index_queues

        @property
        def workers(self):
            return self._workers

        @property
        def workers_status(self):
            return self._workers_status

        @property
        def data_queue(self):
            return self._data_queue

        @property
        def pin_memory_thread_done_event(self):
            if hasattr(self, '_pin_memory_thread_done_event'):
                return self._pin_memory_thread_done_event

        @property
        def pin_memory_thread(self):
            if hasattr(self, '_pin_memory_thread'):
                return self._pin_memory_thread

else:
    class TexarBaseDataLoaderIter(_BaseDataLoaderIter):  # type: ignore
        pass

    class TexarSingleProcessDataLoaderIter(  # type: ignore
        _SingleProcessDataLoaderIter, TexarBaseDataLoaderIter):
        pass

    class TexarMultiProcessingDataLoaderIter(  # type: ignore
        _MultiProcessingDataLoaderIter, TexarBaseDataLoaderIter):
        pass
