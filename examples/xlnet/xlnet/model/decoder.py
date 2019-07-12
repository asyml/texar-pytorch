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

from typing import Dict, List, NamedTuple, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

import texar as tx
from texar.modules import Helper
from xlnet.data.utils import EOD_ID
from xlnet.model.model import XLNet


class XLNetDecoderOutput(NamedTuple):
    logits: Tensor
    r"""A :tensor:`Tensor` of shape ``[batch_size, max_time, vocab_size]``
    containing the logits."""
    sample_id: torch.LongTensor
    r"""A :tensor:`LongTensor` of shape ``[batch_size, max_time]`` containing
    the sampled token indices."""


Output = XLNetDecoderOutput
State = List[Tensor]


class XLNetDecoder(XLNet, tx.modules.DecoderBase[State, Output]):
    def __init__(self, hparams=None):
        super().__init__(hparams)

        self.lm_bias = nn.Parameter(torch.zeros(self._hparams.vocab_size))

    # Variables persistent during decoding.
    _state_cache_len: int
    _state_recompute_memory: bool
    _state_previous_inputs: List[Tensor]  # required for recomputing memory

    @staticmethod
    def _create_input(inputs: List[Tensor], initial: bool = False) \
            -> Dict[str, Tensor]:
        r"""Create input tensors given the list of prompt tokens.
        """
        word_embed = torch.stack(inputs, dim=0)
        seq_len, batch_size, embed_dim = word_embed.size()
        if not initial:
            # Add a dummy token at the end that stands for the token
            # to predict.
            word_embed = torch.cat([
                word_embed,
                word_embed.new_zeros(1, batch_size, embed_dim)
            ], dim=0)
            seq_len += 1
        segment_ids = word_embed.new_zeros(
            seq_len, batch_size, dtype=torch.long)
        if initial:
            target_mapping = None
            permute_mask = None
        else:
            # Only the dummy token is considered target.
            target_mapping = torch.cat([
                torch.zeros(1, seq_len - 1, batch_size),
                torch.ones(1, 1, batch_size)
            ], dim=1).to(device=word_embed.device)
            # Dummy token attends to nothing; actual tokens attend to all.
            permute_mask = torch.cat([
                torch.zeros(seq_len, seq_len - 1, batch_size),
                torch.ones(seq_len, 1, batch_size),
            ], dim=1).to(device=word_embed.device)

        return {
            "word_embed": word_embed,
            "segment_ids": segment_ids,
            "target_mapping": target_mapping,
            "permute_mask": permute_mask,
        }

    def initialize(self,  # pylint: disable=no-self-use
                   helper: Helper, inputs: Optional[Tensor],
                   sequence_length: Optional[torch.LongTensor],
                   initial_state: Optional[State]) \
            -> Tuple[torch.ByteTensor, Tensor, Optional[State]]:
        initial_finished, initial_inputs = helper.initialize(
            inputs, sequence_length)
        return initial_finished, initial_inputs, initial_state

    def step(self, helper: Helper, time: int, inputs: Tensor,
             state: Optional[State]) \
            -> Tuple[Output, Optional[State], Tensor, torch.ByteTensor]:
        self._state_previous_inputs.append(inputs)
        if not self._state_recompute_memory:
            assert state is not None
            net_output, memory = self._forward(
                memory=state, cache_len=self._state_cache_len, two_stream=True,
                **self._create_input(self._state_previous_inputs[-1:]))
            # Omit memory for the dummy token.
            memory = [mem[:-1] for mem in memory]
        else:
            net_output, memory = self._forward(
                two_stream=True,
                **self._create_input(
                    self._state_previous_inputs[-self._state_cache_len:]))
        logits = F.linear(net_output, self.word_embed.weight, self.lm_bias)
        logits = logits[-1]
        sample_ids = helper.sample(time=time, outputs=logits)
        (finished, next_inputs) = helper.next_inputs(
            time=time,
            outputs=logits,
            sample_ids=sample_ids)
        outputs = XLNetDecoderOutput(logits=logits, sample_id=sample_ids)
        return outputs, memory, next_inputs, finished

    def finalize(self, outputs: Output, final_state: Optional[State],
                 sequence_lengths: torch.LongTensor) \
            -> Tuple[Output, Optional[State]]:
        del self._state_cache_len
        del self._state_recompute_memory
        del self._state_previous_inputs
        return super().finalize(outputs, final_state, sequence_lengths)

    def forward(self, start_tokens: torch.LongTensor,
                memory: Optional[State] = None,
                cache_len: int = 512,
                max_decoding_length: Optional[int] = 500,
                recompute_memory: bool = True,
                print_steps: bool = False,
                helper_type: Optional[Union[str, Type[Helper]]] = None,
                **helper_kwargs) \
            -> Tuple[Output, Optional[State]]:
        r"""Perform autoregressive decoding using XLNet. The algorithm is
        largely inspired by: https://github.com/rusiaaman/XLNet-gen.

        Args:
            start_tokens: A LongTensor of shape `(batch_size, prompt_len)`,
                representing the tokenized initial prompt.
            memory (optional): The initial memory.
            cache_len: Length of memory (number of tokens) to cache.
            max_decoding_length (int): Maximum number of tokens to decode.
            recompute_memory (bool): If `True`, the entire memory is recomputed
                for each token to generate. This leads to better performance
                because it enables every generated token to attend to each
                other, compared to reusing previous memory which is equivalent
                to using a causal attention mask. However, it is computationally
                more expensive. Defaults to `True`.
            print_steps (bool): If `True`, will print decoding progress.
            helper: Type (or name of the type) of any sub-class of
                :class:`~texar.modules.decoders.Helper`.
            helper_kwargs: The keyword arguments to pass to constructor of
                the specific helper type.

        :returns: A tuple of `(output, new_memory)`:

            - **`output`**: The sampled tokens as a list of `int`s.
            - **`new_memory`**: The memory of the sampled tokens.
        """

        start_tokens = start_tokens.t()
        self._state_recompute_memory = recompute_memory
        self._state_cache_len = cache_len
        self._state_previous_inputs = list(
            self.word_embed(start_tokens).unbind(dim=0))[:-1]

        if helper_type is None:
            helper_type = tx.modules.SampleEmbeddingHelper

        if not recompute_memory and start_tokens.size(0) > 1:
            _, memory = self._forward(
                memory=memory, cache_len=cache_len,
                **self.create_input(
                    self._state_previous_inputs, initial=True))
        start_tokens = start_tokens[-1]

        helper_kwargs.update(
            embedding=self.word_embed.weight, start_tokens=start_tokens)
        helper_kwargs.setdefault("end_token", EOD_ID)
        helper = tx.utils.get_instance(
            helper_type, helper_kwargs,
            module_paths=['texar.modules.decoders.decoder_helpers'])

        step_hook = None
        if print_steps:
            step_hook = lambda step: print(
                f"\033[2K\rDecoding step: {step}", end='')
        output, new_memory, _ = self.dynamic_decode(
            helper, inputs=None, sequence_length=None, initial_state=memory,
            max_decoding_length=max_decoding_length, step_hook=step_hook)
        if print_steps:
            print("\033[2K\r", end='')

        return output, new_memory
