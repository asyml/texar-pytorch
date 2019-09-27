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
"""Data read/write utilities for Transformer.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

import texar.torch as tx

Example = Tuple[np.ndarray, np.ndarray]


class CustomBatchingStrategy(tx.data.BatchingStrategy[Example]):
    r"""Create dynamically-sized batches for paired text data so that the total
    number of source and target tokens (including padding) inside each batch is
    constrained.

    Args:
        max_tokens (int): The maximum number of source or target tokens inside
            each batch.
    """
    max_src_len: int
    max_tgt_len: int
    cur_batch_size: int

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def reset_batch(self) -> None:
        self.max_src_len = 0
        self.max_tgt_len = 0
        self.cur_batch_size = 0

    def add_example(self, ex: Example) -> bool:
        max_src_len = max(self.max_src_len, len(ex[0]))
        max_tgt_len = max(self.max_tgt_len, len(ex[1]))
        if ((self.cur_batch_size + 1) *
                max(max_src_len, max_tgt_len) > self.max_tokens):
            return False
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.cur_batch_size += 1
        return True


class Seq2SeqData(tx.data.DatasetBase[Example, Example]):
    r"""A dataset that reads processed paired text from dumped NumPy files.

    Args:
        filename (str): The path to the dumped NumPy file.
        hparams: A `dict` or instance of :class:`~texar.HParams` containing
            hyperparameters. See :meth:`default_hparams` for the defaults.
        device: The device of the produces batches. For GPU training, set to
            current CUDA device.
    """

    def __init__(self, filename: str, hparams=None,
                 device: Optional[torch.device] = None):
        data: List[Example] = np.load(
            filename,
            encoding="latin1",
            allow_pickle=True).tolist()
        source = tx.data.SequenceDataSource(data)
        super().__init__(source, hparams, device)

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
        }

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        # Add EOS tokens.
        src_seqs = [ex[0].tolist() + [self._hparams.eos_id] for ex in examples]
        tgt_seqs = [ex[1].tolist() + [self._hparams.eos_id] for ex in examples]
        # Pad sentences to equal length.
        source, _ = tx.data.padded_batch(
            src_seqs, pad_value=self._hparams.pad_id)
        target_output, _ = tx.data.padded_batch(
            tgt_seqs, pad_value=self._hparams.pad_id)
        # Add BOS token to the target inputs.
        target_input = np.pad(
            target_output[:, :-1], ((0, 0), (1, 0)),
            mode="constant", constant_values=self._hparams.bos_id)
        source, target_input, target_output = [
            torch.from_numpy(x)
            for x in [source, target_input, target_output]
        ]
        return tx.data.Batch(
            len(examples),
            source=source,
            target_input=target_input,
            target_output=target_output,
        )
