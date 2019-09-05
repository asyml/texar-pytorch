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
Base class for RNN decoders.
"""

from typing import Optional, Tuple, TypeVar

import torch
from torch import nn

from texar.torch.core import layers
from texar.torch.core.cell_wrappers import RNNCellBase
from texar.torch.modules.decoders import decoder_helpers
from texar.torch.modules.decoders.decoder_base import (
    DecoderBase, TokenEmbedder, TokenPosEmbedder, _make_output_layer)
from texar.torch.modules.decoders.decoder_helpers import Helper
from texar.torch.utils import utils

__all__ = [
    'RNNDecoderBase',
]

State = TypeVar('State')
Output = TypeVar('Output')  # output type can be of any nested structure


class RNNDecoderBase(DecoderBase[State, Output]):
    r"""Base class inherited by all RNN decoder classes.
    See :class:`~texar.torch.modules.BasicRNNDecoder` for the arguments.

    See :meth:`forward` for the inputs and outputs of RNN decoders in general.
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 token_embedder: Optional[TokenEmbedder] = None,
                 token_pos_embedder: Optional[TokenPosEmbedder] = None,
                 cell: Optional[RNNCellBase] = None,
                 output_layer: Optional[nn.Module] = None,
                 input_time_major: bool = False,
                 output_time_major: bool = False,
                 hparams=None):
        super().__init__(token_embedder, token_pos_embedder,
                         input_time_major, output_time_major, hparams=hparams)

        self._input_size = input_size
        self._vocab_size = vocab_size

        # Make RNN cell
        self._cell = cell or layers.get_rnn_cell(
            input_size, self._hparams.rnn_cell)
        self._beam_search_cell = None

        # Make the output layer
        self._output_layer, _ = _make_output_layer(
            output_layer, self._vocab_size, self._cell.hidden_size,
            self._hparams.output_layer_bias)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        The hyperparameters are the same as in
        :meth:`~texar.torch.modules.BasicRNNDecoder.default_hparams` of
        :class:`~texar.torch.modules.BasicRNNDecoder`, except that the default
        ``"name"`` here is ``"rnn_decoder"``.
        """
        return {
            'rnn_cell': layers.default_rnn_cell_hparams(),
            'helper_train': decoder_helpers.default_helper_train_hparams(),
            'helper_infer': decoder_helpers.default_helper_infer_hparams(),
            'max_decoding_length_train': None,
            'max_decoding_length_infer': None,
            'name': 'rnn_decoder',
            "output_layer_bias": True,
        }

    def forward(self,  # type: ignore
                inputs: Optional[torch.Tensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                initial_state: Optional[State] = None,
                helper: Optional[Helper] = None,
                max_decoding_length: Optional[int] = None,
                impute_finished: bool = False,
                infer_mode: Optional[bool] = None, **kwargs) \
            -> Tuple[Output, Optional[State], torch.LongTensor]:
        r"""Performs decoding. This is a shared interface for both
        :class:`~texar.torch.modules.BasicRNNDecoder` and
        :class:`~texar.torch.modules.AttentionRNNDecoder`.

        Implementation calls :meth:`initialize` once and :meth:`step`
        repeatedly on the decoder object. Please refer to
        `tf.contrib.seq2seq.dynamic_decode
        <https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode>`_.

        See Also:
            Arguments of :meth:`create_helper`.

        Args:
            inputs (optional): Input tensors for teacher forcing decoding.
                Used when :attr:`decoding_strategy` is set to
                ``"train_greedy"``, or when `hparams`-configured helper is used.

                The attr:`inputs` is a :tensor:`LongTensor` used as index to
                look up embeddings and feed in the decoder. For example, if
                :attr:`embedder` is an instance of
                :class:`~texar.torch.modules.WordEmbedder`, then :attr:`inputs`
                is usually a 2D int Tensor `[batch_size, max_time]` (or
                `[max_time, batch_size]` if `input_time_major` == `True`)
                containing the token indexes.
            sequence_length (optional): A 1D int Tensor containing the
                sequence length of :attr:`inputs`.
                Used when `decoding_strategy="train_greedy"` or
                `hparams`-configured helper is used.
            initial_state (optional): Initial state of decoding.
                If `None` (default), zero state is used.
            max_decoding_length: A int scalar Tensor indicating the maximum
                allowed number of decoding steps. If `None` (default), either
                `hparams["max_decoding_length_train"]` or
                `hparams["max_decoding_length_infer"]` is used
                according to :attr:`mode`.
            impute_finished (bool): If `True`, then states for batch
                entries which are marked as finished get copied through and
                the corresponding outputs get zeroed out.  This causes some
                slowdown at each time step, but ensures that the final state
                and outputs have the correct values and that backprop ignores
                time steps that were marked as finished.
            helper (optional): An instance of
                :class:`~texar.torch.modules.Helper`
                that defines the decoding strategy. If given,
                ``decoding_strategy`` and helper configurations in
                :attr:`hparams` are ignored.
            infer_mode (optional): If not `None`, overrides mode given by
                `self.training`.
            **kwargs: Other keyword arguments for constructing helpers
                defined by ``hparams["helper_train"]`` or
                ``hparams["helper_infer"]``.

        Returns:
            ``(outputs, final_state, sequence_lengths)``, where

            - `outputs`: an object containing the decoder output on all
              time steps.
            - `final_state`: the cell state of the final time step.
            - `sequence_lengths`: a :tensor:`LongTensor` of shape
              ``[batch_size]`` containing the length of each sample.
        """
        # TODO: Add faster code path for teacher-forcing training.

        # Helper
        if helper is None:
            helper = self._create_or_get_helper(infer_mode, **kwargs)

        if (isinstance(helper, decoder_helpers.TrainingHelper) and
                (inputs is None or sequence_length is None)):
            raise ValueError("'input' and 'sequence_length' must not be None "
                             "when using 'TrainingHelper'.")

        # Initial state
        self._cell.init_batch()

        # Maximum decoding length
        if max_decoding_length is None:
            if self.training:
                max_decoding_length = self._hparams.max_decoding_length_train
            else:
                max_decoding_length = self._hparams.max_decoding_length_infer
            if max_decoding_length is None:
                max_decoding_length = utils.MAX_SEQ_LENGTH

        return self.dynamic_decode(
            helper, inputs, sequence_length, initial_state,
            max_decoding_length, impute_finished)

    def _get_beam_search_cell(self):
        self._beam_search_cell = self._cell
        return self._cell

    @property
    def output_size(self):
        r"""Output size of one step.
        """
        raise NotImplementedError

    def initialize(self, helper: Helper, inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor],
                   initial_state: Optional[State]) \
            -> Tuple[torch.ByteTensor, torch.Tensor, Optional[State]]:
        initial_finished, initial_inputs = helper.initialize(
            self.embed_tokens, inputs, sequence_length)
        if initial_state is None:
            state = self._cell.init_batch()
        else:
            state = initial_state
        return (initial_finished, initial_inputs, state)

    def step(self, helper: Helper, time: int,
             inputs: torch.Tensor, state: Optional[State]) \
            -> Tuple[Output, State]:
        raise NotImplementedError

    def next_inputs(self, helper: Helper, time: int, outputs: Output) -> \
            Tuple[torch.Tensor, torch.ByteTensor]:
        raise NotImplementedError

    @property
    def cell(self):
        r"""The RNN cell.
        """
        return self._cell

    def zero_state(self, batch_size):
        r"""Zero state of the RNN cell.
        Equivalent to :attr:`decoder.cell.zero_state`.
        """
        return self._cell.zero_state(batch_size=batch_size)

    @property
    def state_size(self):
        r"""The state size of decoder cell.
        Equivalent to :attr:`decoder.cell.state_size`.
        """
        return self._cell.hidden_size

    @property
    def output_layer(self):
        r"""The output layer.
        """
        return self._output_layer
