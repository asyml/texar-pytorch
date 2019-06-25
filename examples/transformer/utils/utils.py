# Copyright 2018 The Texar Authors. All Rights Reserved.
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
Helper functions for model training.
"""

import random
import math
import logging
import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def batch_size_fn(new, count, size_so_far):  # pylint: disable=unused-argument
    if count == 1 or not hasattr(batch_size_fn, 'max_src_in_batch'):
        batch_size_fn.max_src_in_batch = 0
        batch_size_fn.max_tgt_in_batch = 0
    batch_size_fn.max_src_in_batch = max(
        batch_size_fn.max_src_in_batch, len(new[0]) + 1)
    batch_size_fn.max_tgt_in_batch = max(
        batch_size_fn.max_tgt_in_batch, len(new[1]) + 1)
    return count * max(batch_size_fn.max_src_in_batch,
                       batch_size_fn.max_tgt_in_batch)


class CustomBatchingStrategy(tx.data.BatchingStrategy):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.sum_src_tokens = 0
        self.sum_tgt_tokens = 0

    def reset_batch(self) -> None:
        self.sum_src_tokens = 0
        self.sum_tgt_tokens = 0

    def add_example(self, ex) -> bool:
        src_len = len(ex['source_text'])
        tgt_len = len(ex['target_text'])
        if (src_len + self.sum_src_tokens > self.max_tokens or
                tgt_len + self.sum_tgt_tokens > self.max_tokens):
            return False
        self.sum_src_tokens += src_len
        self.sum_tgt_tokens += tgt_len
        return True


def get_lr_multiplier(step: int, warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    square-root decay.
    """
    multiplier = (min(1.0, step / warmup_steps) *
                  (1 / math.sqrt(max(step, warmup_steps))))
    return multiplier


def get_logger(log_path):
    """Returns a logger.

    Args:
        log_path (str): Path to the log file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
    logger.addHandler(fh)
    return logger


def list_strip_eos(list_, eos_token):
    """Strips EOS token from a list of lists of tokens.
    """
    list_strip = []
    for elem in list_:
        if eos_token in elem:
            elem = elem[:elem.index(eos_token)]
        list_strip.append(elem)
    return list_strip
