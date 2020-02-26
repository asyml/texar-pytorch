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
Various sampler classes.
"""

# pylint: disable=protected-access

from typing import (
    Any, Callable, Generic, Iterator, List, Optional, Tuple, TypeVar, Union)

import torch
from torch.utils.data import sampler as torch_sampler

from texar.torch.data.data.data_base import DatasetBase


__all__ = [
    "BatchingStrategy",
    "TokenCountBatchingStrategy",
]

Example = TypeVar('Example')


# pylint: disable=attribute-defined-outside-init
# TODO: Remove this when Pylint fixes the bug. If the `disable` directive is not
#  added, Pylint incorrectly reports this error for `self.size` in subclasses of
#  `SamplerBase` in Python 3.6 due to use of the Generic class.
#  See Pylint issue: https://github.com/PyCQA/pylint/issues/2981

class SamplerBase(torch_sampler.Sampler, Generic[Example]):
    r"""A subclass of :torch_docs:`~torch.utils.data.Sampler
    <data.html#torch.utils.data.Sampler>` that supports:

    - Returning raw examples when required.
    - Creating iterators with unknown dataset size.

    This class is used internally in
    :class:`~texar.torch.data.data.DataIterator`. It calls the
    :meth:`~texar.torch.data.data.DatasetBase._prefetch_source` method to ensure
    the required number of raw examples are prefetched from source.

    Args:
        data: The :class:`~texar.torch.data.data.DatasetBase` instance.
    """
    size: Optional[int]

    def __init__(self, data: DatasetBase[Any, Example]):
        super().__init__(data)

        self._data = data
        self.size = None

    def _iterator_given_size(self, size: int) -> Iterator[int]:
        r"""Return an iterator that generates samples when the dataset size
        is given.

        Args:
            size: The dataset size.
        """
        raise NotImplementedError

    def _iterator_unknown_size(self) -> Iterator[int]:
        r"""Return an iterator that generates samples when the dataset size
        is unknown. This iterator must also call
        :meth:`texar.torch.data.data.DatasetBase._prefetch_source` and check
        whether the dataset size can be determined, before yielding the index.
        See example implementations for details.
        """
        raise NotImplementedError

    def __iter__(self) -> Union[Iterator[int], Iterator[Tuple[int, Example]]]:
        r"""Return an iterator based on the dataset settings.
        """
        self.size = self._data._dataset_size
        if (not self._data._fully_cached or
                self._data._should_call_prefetch_source):
            self._data._start_iteration()
            # First epoch of lazy loading, calling prefetch, and returning
            # indices and examples.
            iterator = self._iterator_unknown_size()
        else:
            # Non-lazy loading, or when dataset has been fully iterated.
            assert self.size is not None
            iterator = self._iterator_given_size(self.size)

        if self._data._should_call_prefetch_processed:
            # Processing routine is performed in main process. Yield
            # processed examples instead.
            map_fn = lambda idx: (idx, self._data._processed_cache[idx])
        elif self._data._should_yield_raw_example:
            # Return indices and examples for any epoch in this case.
            map_fn = lambda idx: (idx, self._data._source[idx])
        else:
            map_fn = None  # type: ignore
        if map_fn is not None:
            return map(map_fn, iterator)

        return iterator

    def __len__(self):
        if self.size is not None:
            return self.size
        raise AttributeError("Dataset size cannot be determined at this point")


class SequentialSampler(SamplerBase[Example]):
    r"""Samples elements sequentially, always in the same order. Same as
    :torch_docs:`~torch.utils.data.SequentialSampler
    <data.html#torch.utils.data.SequentialSampler>`
    """

    def _iterator_given_size(self, size: int) -> Iterator[int]:
        return iter(range(size))

    def _iterator_unknown_size(self) -> Iterator[int]:
        index = 0
        while True:
            cur_size = self._data._prefetch_source(index)
            if cur_size is not None:
                self.size = cur_size
                break
            yield index
            index += 1


class RandomSampler(SamplerBase[Example]):
    r"""Samples elements randomly. If without replacement, then sample from a
    shuffled dataset. If with replacement, then user can specify ``num_samples``
    to draw.

    This class uses :torch_docs:`torch.utils.data.RandomSampler
    <data.html#torch.utils.data.RandomSampler>` directly. Given the
    nature of such shuffling, it cannot be used for iterators with unknown size.

    Args:
        data: The :class:`~texar.torch.data.data.DatasetBase` instance.
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if `True`,
            default=False
    """

    def __init__(self, data: DatasetBase[Any, Example],
                 replacement: bool = False, num_samples: Optional[int] = None):
        super().__init__(data)
        self._sampler = torch_sampler.RandomSampler(
            data, replacement, num_samples)

    def _iterator_given_size(self, size: int) -> Iterator[int]:
        del size  # not used
        return iter(self._sampler)

    def _iterator_unknown_size(self) -> Iterator[int]:
        raise TypeError(
            "RandomSampler does not support lazy data loading. To perform "
            "shuffling with lazy loading, use BufferShuffleSampler.")


class BufferShuffleSampler(SamplerBase[Example]):
    r"""A :torch_docs:`~torch.utils.data.Sampler
    <data.html#torch.utils.data.Sampler>` that uses a shuffle buffer, as
    in TensorFlow. The buffer is first filled with data examples. Each time a
    sample is drawn from the buffer, and the drawn sample is replaced with the
    next data example.

    This class is used internally in
    :class:`~texar.torch.data.data.DataIterator`. It calls the
    :meth:`~texar.torch.data.data.DatasetBase._prefetch_source` method to ensure
    the required number of raw examples are prefetched from source.

    Args:
        data: The :class:`~texar.torch.data.data.DatasetBase` instance.
        buffer_size: The size of the shuffle buffer. Use larger buffer sizes for
            more uniformly-random shuffling.
    """

    def __init__(self, data: DatasetBase[Any, Example], buffer_size: int):
        super().__init__(data)
        self.buffer_size = buffer_size

    def _iterator_given_size(self, size) -> Iterator[int]:
        if self.buffer_size >= size:
            yield from iter(torch.randperm(size).tolist())
            return

        buffer = list(range(self.buffer_size))
        for x in range(self.buffer_size, size):
            sample = torch.randint(self.buffer_size, (1,)).item()
            index = buffer[sample]
            yield index
            buffer[sample] = x
        yield from (buffer[x] for x in torch.randperm(self.buffer_size))

    def _iterator_unknown_size(self) -> Iterator[int]:
        buffer = list(range(self.buffer_size))
        x = self.buffer_size
        while True:
            sample = torch.randint(self.buffer_size, (1,)).item()
            index = buffer[sample]
            cur_size = self._data._prefetch_source(index)
            if cur_size is not None and index >= cur_size:
                self.size = cur_size
            if self.size is not None and index >= self.size:
                break
            yield index
            buffer[sample] = x
            x += 1
        yield from (buffer[x] for x in torch.randperm(self.buffer_size)
                    if buffer[x] < self.size)


# pylint: enable=attribute-defined-outside-init


class BatchingStrategy(Generic[Example]):
    r"""Decides batch boundaries in dynamic batching. Please refer to
    :class:`TokenCountBatchingStrategy` for a concrete example.
    """

    def reset_batch(self) -> None:
        r"""Reset the internal state of the batching strategy. This method is
        called at the start of iteration, and after each batch is yielded.
        """
        raise NotImplementedError

    def add_example(self, example: Example) -> bool:
        r"""Add an example into the current batch, and modify internal states
        accordingly. If the example should not be added to the batch, this
        method does not modify the internal state, and returns `False`.

        Args:
            example: The example to add to the batch.

        Returns:
            A boolean value indicating whether :attr:`example` should be added
            to the batch.
        """
        raise NotImplementedError


class TokenCountBatchingStrategy(BatchingStrategy[Example]):
    r"""Create dynamically-sized batches so that the total number of tokens
    inside each batch is constrained.

    Args:
        max_tokens (int): The maximum number of tokens inside each batch.
        max_batch_size (int, optional): The maximum number of examples for each
            batch. If `None`, batches can contain arbitrary number of examples
            as long as the total number of tokens does not exceed
            :attr:`max_tokens`.
        length_fn (callable, optional): A function taking a data example as
            argument, and returning the number of tokens in the example. By
            default, :python:`len` is used, which is the desired behavior if the
            dataset in question is a :class:`~texar.torch.data.MonoTextData`.
    """
    sum_tokens: int
    cur_batch_size: int

    def __init__(self, max_tokens: int, max_batch_size: Optional[int] = None,
                 length_fn: Optional[Callable[[Example], int]] = None):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.length_fn: Callable[[Example], int]
        self.length_fn = length_fn or len  # type: ignore

    def reset_batch(self) -> None:
        self.sum_tokens = 0
        self.cur_batch_size = 0

    def add_example(self, example: Example) -> bool:
        if self.cur_batch_size == self.max_batch_size:
            return False
        cur_tokens = self.length_fn(example)
        if cur_tokens + self.sum_tokens > self.max_tokens:
            return False

        self.cur_batch_size += 1
        self.sum_tokens += cur_tokens
        return True


class DynamicBatchSampler(torch_sampler.BatchSampler, Generic[Example]):
    r"""A subclass of :torch_docs:`~torch.utils.data.BatchSampler
    <data.html#torch.utils.data.BatchSampler>` that supports dynamic batching
    through a user-provided :class:`BatchingStrategy`. This class is used
    internally.

    Args:
        dataset: The dataset to create batches from.
        sampler: An instance of :class:`SamplerBase` that returns indices of
            each sampled example.
        strategy: An instance of :class:`BatchingStrategy` that decides whether
            a batch should be yielded.
    """

    def __init__(self, dataset: DatasetBase[Any, Example],  # pylint: disable=super-init-not-called
                 sampler: SamplerBase, strategy: BatchingStrategy[Example]):
        self.dataset = dataset
        self.sampler = sampler
        self.strategy = strategy

    def __iter__(self) -> Union[Iterator[List[int]],  # type: ignore
                                Iterator[List[Tuple[int, Example]]]]:
        batch = []  # type: ignore
        self.strategy.reset_batch()
        for idx in self.sampler:
            if isinstance(idx, tuple):
                example = self.dataset[idx[0]]
            else:
                example = self.dataset[idx]
            while not self.strategy.add_example(example):
                if len(batch) == 0:
                    raise ValueError(f"Batching strategy refused to add "
                                     f"example {idx} to empty batch.")
                yield batch
                batch = []
                self.strategy.reset_batch()
            batch.append(idx)
        if len(batch) > 0:
            yield batch
            self.strategy.reset_batch()

    def __len__(self):
        raise TypeError("DynamicBatchSampler does not support __len__")
