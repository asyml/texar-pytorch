import contextlib
import resource
import time
import unittest
from typing import List, Optional, Tuple

import gc
import numpy as np
import torch

from texar.torch.data.data.data_base import DatasetBase, DataSource
from texar.torch.data.data.data_iterators import DataIterator
from texar.torch.data.data.dataset_utils import Batch
from texar.torch.data.data.text_data_base import TextLineDataSource
from texar.torch.data.vocabulary import Vocab
from texar.torch.utils.test import data_test
from texar.torch.utils.utils import AnyDict


@contextlib.contextmanager
def work_in_progress(msg):
    print(msg + "... ", flush=True)
    begin_time = time.time()
    yield
    time_consumed = time.time() - begin_time
    print(f"done. ({time_consumed:.2f}s)", flush=True)


RawExample = List[str]
Example = Tuple[np.ndarray, np.ndarray]


class ParallelData(DatasetBase[RawExample, Example]):
    def __init__(self, source: DataSource[RawExample],
                 src_vocab_path: str,
                 tgt_vocab_path: str,
                 hparams: AnyDict,
                 device: Optional[torch.device] = None):
        # hparams.update(parallelize_processing=False)
        self.src_vocab = Vocab(src_vocab_path)
        self.tgt_vocab = Vocab(tgt_vocab_path)
        self.device = device
        super().__init__(source, hparams=hparams)

    def process(self, raw_example: RawExample) -> Example:
        src, tgt = raw_example
        src = self.src_vocab.map_tokens_to_ids_py(src.split())
        tgt = self.tgt_vocab.map_tokens_to_ids_py(tgt.split())
        return src, tgt

    def collate(self, examples: List[Example]) -> Batch:
        src_pad_length = max(len(src) for src, _ in examples)
        tgt_pad_length = max(len(tgt) for _, tgt in examples)
        batch_size = len(examples)
        src_indices = np.zeros((batch_size, src_pad_length), dtype=np.int64)
        tgt_indices = np.zeros((batch_size, tgt_pad_length), dtype=np.int64)
        for b_idx, (src, tgt) in enumerate(examples):
            src_indices[b_idx, :len(src)] = src
            tgt_indices[b_idx, :len(tgt)] = tgt
        src_indices = torch.from_numpy(src_indices)
        tgt_indices = torch.from_numpy(tgt_indices)
        return Batch(batch_size, src=src_indices, tgt=tgt_indices)


def wrap_progress(func):
    from tqdm import tqdm
    return lambda: tqdm(func(), leave=False)


def get_process_memory():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


@data_test
class LargeFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = TextLineDataSource(
            '../../Downloads/en-es.bicleaner07.txt.gz',
            compression_type='gzip', delimiter='\t')
        self.source.__iter__ = wrap_progress(  # type: ignore
            self.source.__iter__)
        self.num_workers = 3
        self.batch_size = 64

    def _test_modes_with_workers(self, lazy_mode: str, cache_mode: str,
                                 num_workers: int):
        from tqdm import tqdm
        gc.collect()
        mem = get_process_memory()
        with work_in_progress(f"Data loading with lazy mode '{lazy_mode}' "
                              f"and cache mode '{cache_mode}' "
                              f"with {num_workers} workers"):
            print(f"Memory before: {mem:.2f} MB")
            with work_in_progress("Construction"):
                data = ParallelData(self.source,
                                    '../../Downloads/src.vocab',
                                    '../../Downloads/tgt.vocab',
                                    {'batch_size': self.batch_size,
                                     'lazy_strategy': lazy_mode,
                                     'cache_strategy': cache_mode,
                                     'num_parallel_calls': num_workers,
                                     'shuffle': False,
                                     'allow_smaller_final_batch': False,
                                     'max_dataset_size': 100000})
            print(f"Memory after construction: {mem:.2f} MB")
            iterator = DataIterator(data)
            with work_in_progress("Iteration"):
                for batch in tqdm(iterator, leave=False):
                    self.assertEqual(batch.batch_size, self.batch_size)
            gc.collect()
            print(f"Memory after iteration: {mem:.2f} MB")
            with work_in_progress("2nd iteration"):
                for batch in tqdm(iterator, leave=False):
                    self.assertEqual(batch.batch_size, self.batch_size)

    def _test_modes(self, lazy_mode: str, cache_mode: str):
        self._test_modes_with_workers(lazy_mode, cache_mode, self.num_workers)
        self._test_modes_with_workers(lazy_mode, cache_mode, 1)

    def test_none_processed(self):
        self._test_modes('none', 'processed')

    def test_process_loaded(self):
        self._test_modes('process', 'loaded')

    def test_process_processed(self):
        self._test_modes('process', 'processed')

    def test_all_none(self):
        self._test_modes('all', 'none')

    def test_all_loaded(self):
        self._test_modes('all', 'loaded')

    def test_all_processed(self):
        self._test_modes('all', 'processed')

    def _test_all_combinations(self):
        self.test_none_processed()
        self.test_process_loaded()
        self.test_process_processed()
        self.test_all_none()
        self.test_all_loaded()
        self.test_all_processed()
