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

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import torch
from torch.nn import functional as F

from texar.torch.modules.decoders.decoder_base import DecoderBase
from texar.torch.modules.decoders.decoder_helpers import (
    Helper, SampleEmbeddingHelper)
from texar.torch.modules.encoders.xlnet_encoder import XLNetEncoder
from texar.torch.utils import get_instance

__all__ = [
    'XLNetDecoderOutput',
    'XLNetDecoder',
]


class XLNetDecoderOutput(NamedTuple):
    r"""The output of :class:`XLNetDecoder`.
    """
    logits: torch.Tensor
    r"""A :tensor:`Tensor` of shape ``[batch_size, max_time, vocab_size]``
    containing the logits."""
    sample_id: torch.LongTensor
    r"""A :tensor:`LongTensor` of shape ``[batch_size, max_time]``
    (or ``[batch_size, max_time, vocab_size]``) containing the sampled token
    indices. Note that the shape of ``sample_id`` is different for different
    decoding strategy or helper. Please refer to
    :class:`~texar.torch.modules.Helper` for the detailed information."""


Output = XLNetDecoderOutput
State = List[torch.Tensor]


class XLNetDecoder(XLNetEncoder, DecoderBase[Optional[State], Output]):
    r"""Raw XLNet module for decoding sequences. Please see
    :class:`~texar.torch.modules.PretrainedXLNetMixin` for a brief description
    of XLNet.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``xlnet-based-cased``). Please refer to
            :class:`~texar.torch.modules.PretrainedXLNetMixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """
    _IS_DECODE = True
    # Variables persistent during decoding.
    _state_cache_len: int
    _state_recompute_memory: bool
    # required for recomputing memory
    _state_previous_inputs: List[torch.Tensor]

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        * The decoder arch is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the decoder arch is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the decoder arch is defined by the
          configurations in `hparams` and weights are randomly initialized.

        .. code-block:: python

            {
                "pretrained_model_name": "xlnet-base-cased",
                "untie_r": True,
                "num_layers": 12,
                "mem_len": 0,
                "reuse_len": 0,
                "num_heads": 12,
                "hidden_dim": 768,
                "head_dim": 64,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "use_segments": True,
                "ffn_inner_dim": 3072,
                "activation": 'gelu',
                "vocab_size": 32000,
                "max_seq_length": 512,
                "initializer": None,
                "name": "xlnet_decoder",
            }

        Here:

        The default parameters are values for cased XLNet-Base model.

        `"pretrained_model_name"`: str or None
            The name of the pre-trained XLNet model. If None, the model
            will be randomly initialized.

        `"untie_r"`: bool
            Whether to untie the biases in attention.

        `"num_layers"`: int
            The number of stacked layers.

        `"mem_len"`: int
            The number of tokens to cache.

        `"reuse_len"`: int
            The number of tokens in the current batch to be cached and reused
            in the future.

        `"num_heads"`: int
            The number of attention heads.

        `"hidden_dim"`: int
            The hidden size.

        `"head_dim"`: int
            The dimension size of each attention head.

        `"dropout"`: float
            Dropout rate.

        `"attention_dropout"`: float
            Dropout rate on attention probabilities.

        `"use_segments"`: bool
            Whether to use segment embedding.

        `"ffn_inner_dim"`: int
            The hidden size in feed-forward layers.

        `"activation"`: str
            `relu` or `gelu`.

        `"vocab_size"`: int
            The vocabulary size.

        `"max_seq_length"`: int
            The maximum sequence length for `RelativePositionalEncoding`.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.torch.core.get_initializer` for details.

        `"name"`: str
            Name of the module.
        """
        return {
            'pretrained_model_name': 'xlnet-base-cased',
            'untie_r': True,
            'num_layers': 12,
            'mem_len': 0,
            'reuse_len': 0,
            # layer
            'num_heads': 12,
            'hidden_dim': 768,
            'head_dim': 64,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'use_segments': True,
            # ffn
            'ffn_inner_dim': 3072,
            'activation': 'gelu',
            # embedding
            'vocab_size': 32000,
            'max_seq_length': 512,
            'initializer': None,
            'name': "xlnet_decoder",
            '@no_typecheck': ['pretrained_model_name'],
        }

    @staticmethod
    def _create_input(inputs: List[torch.Tensor],
                      initial: bool = False) \
            -> Dict[str, torch.Tensor]:
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
        return_dict = {
            "word_embed": word_embed.permute(1, 0, 2),
            "segment_ids": segment_ids.permute(1, 0),
        }

        if not initial:
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
            return_dict.update({
                "target_mapping": target_mapping.permute(2, 0, 1),
                "permute_mask": permute_mask.permute(2, 0, 1),
            })

        return return_dict

    def embed_tokens(self, tokens: torch.LongTensor,
                     positions: torch.LongTensor) -> torch.Tensor:  # pylint: disable=unused-argument
        return self.word_embed(tokens)

    def initialize(self,
                   helper: Helper,
                   inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor],
                   initial_state: Optional[State]) \
            -> Tuple[torch.ByteTensor, torch.Tensor, Optional[State]]:
        initial_finished, initial_inputs = helper.initialize(
            self.embed_tokens, inputs, sequence_length)
        return initial_finished, initial_inputs, initial_state

    def step(self, helper: Helper, time: int, inputs: torch.Tensor,
             state: Optional[State]) -> \
            Tuple[Output, Optional[State]]:
        self._state_previous_inputs.append(inputs)
        if self._state_recompute_memory:
            net_output, memory = self._forward(
                two_stream=True,
                **self._create_input(
                    self._state_previous_inputs[-self._state_cache_len:]))
        else:
            assert state is not None
            net_output, memory = self._forward(
                memory=state, cache_len=self._state_cache_len, two_stream=True,
                **self._create_input(self._state_previous_inputs[-1:]))
            assert memory is not None
            # Omit memory for the dummy token.
            memory = [mem[:, :-1] for mem in memory]

        logits = F.linear(net_output, self.word_embed.weight, self.lm_bias)
        logits = logits[:, -1]
        sample_ids = helper.sample(time=time, outputs=logits)
        outputs = XLNetDecoderOutput(logits=logits, sample_id=sample_ids)
        return outputs, memory

    def next_inputs(self, helper: Helper, time: int,
                    outputs: Output) -> \
            Tuple[torch.Tensor, torch.ByteTensor]:
        finished, next_inputs = helper.next_inputs(
            self.embed_tokens, time, outputs.logits, outputs.sample_id)
        return next_inputs, finished

    def finalize(self, outputs, final_state, sequence_lengths):
        del self._state_cache_len
        del self._state_recompute_memory
        del self._state_previous_inputs
        return super().finalize(outputs, final_state, sequence_lengths)

    def forward(self,  # type: ignore
                start_tokens: torch.LongTensor,
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
            start_tokens: A LongTensor of shape `[batch_size, prompt_len]`,
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
                :class:`~texar.torch.modules.Helper`.
            helper_kwargs: The keyword arguments to pass to constructor of
                the specific helper type.

        :returns: A tuple of `(output, new_memory)`:
            - **`output`**: The sampled tokens as a list of integers.
            - **`new_memory`**: The memory of the sampled tokens.
        """

        start_tokens = start_tokens.t()
        self._state_recompute_memory = recompute_memory
        self._state_cache_len = cache_len
        self._state_previous_inputs = list(
            self.word_embed(start_tokens).unbind(dim=0))[:-1]

        if helper_type is None:
            helper_type = SampleEmbeddingHelper

        if not recompute_memory and start_tokens.size(0) > 1:
            _, memory = self._forward(
                memory=memory, cache_len=cache_len,
                **self._create_input(
                    self._state_previous_inputs, initial=True))
        start_tokens = start_tokens[-1]

        helper_kwargs.update(start_tokens=start_tokens)

        if helper_kwargs.get("end_token") is None:
            raise ValueError("'end_token' must be specified.")

        helper = get_instance(
            helper_type, helper_kwargs,
            module_paths=['texar.torch.modules.decoders.decoder_helpers'])

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
