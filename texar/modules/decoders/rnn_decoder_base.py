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

# pylint: disable=too-many-arguments, no-name-in-module, too-many-statements
# pylint: disable=too-many-branches, protected-access, too-many-locals
# pylint: disable=arguments-differ, too-many-instance-attributes

from typing import Optional, Tuple, TypeVar

import torch
from torch import nn

from texar import HParams
from texar.core import RNNCellBase, layers
from texar.core.cell_wrappers import HiddenState
from texar.modules.decoders import decoder_helpers
from texar.modules.decoders.decoder_base import DecoderBase
from texar.modules.decoders.decoder_helpers import Helper
from texar.utils import utils

__all__ = [
    'RNNDecoderBase',
]

Output = TypeVar('Output')  # output type can be of any nested structure


class RNNDecoderBase(DecoderBase[HiddenState, Output]):
    r"""Base class inherited by all RNN decoder classes.
    See :class:`~texar.modules.BasicRNNDecoder` for the arguments.

    See :meth:`_build` for the inputs and outputs of RNN decoders in general.
    """

    def __init__(self,
                 cell: Optional[RNNCellBase] = None,
                 vocab_size: Optional[int] = None,
                 output_layer: Optional[nn.Module] = None,
                 input_size: Optional[int] = None,
                 input_time_major: bool = False,
                 output_time_major: bool = False,
                 hparams: Optional[HParams] = None):
        super().__init__(vocab_size, input_size, input_time_major,
                         output_time_major, hparams)

        # Make RNN cell
        self._cell = cell or layers.get_rnn_cell(
            input_size, self._hparams.rnn_cell)
        self._beam_search_cell = None

        # Make the output layer
        if output_layer is not None:
            if (output_layer is not layers.identity and
                    not isinstance(output_layer, nn.Module)):
                raise ValueError(
                    "`output_layer` must be either `texar.core.identity` or "
                    "an instance of `nn.Module`.")
            self._output_layer = output_layer
        elif self._vocab_size is not None:
            self._output_layer = nn.Linear(
                self._cell.hidden_size, self._vocab_size)
        else:
            raise ValueError(
                "Either `output_layer` or `vocab_size` must be provided. "
                "Set `output_layer=texar.core.identity` if no output layer "
                "is desired.")

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        The hyperparameters are the same as in
        :meth:`~texar.modules.BasicRNNDecoder.default_hparams` of
        :class:`~texar.modules.BasicRNNDecoder`, except that the default
        "name" here is "rnn_decoder".
        """
        return {
            'rnn_cell': layers.default_rnn_cell_hparams(),
            'helper_train': decoder_helpers.default_helper_train_hparams(),
            'helper_infer': decoder_helpers.default_helper_infer_hparams(),
            'max_decoding_length_train': None,
            'max_decoding_length_infer': None,
            'name': 'rnn_decoder',
        }

    @classmethod
    def _get_batch_size_from_state(cls, state: HiddenState) -> int:
        if isinstance(state, (list, tuple)):
            return cls._get_batch_size_from_state(state[0])
        return state.size(0)

    def forward(self,  # type: ignore
                inputs: Optional[torch.Tensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                initial_state: Optional[HiddenState] = None,
                helper: Optional[Helper] = None,
                max_decoding_length: Optional[int] = None,
                impute_finished: bool = False,
                infer_mode: Optional[bool] = None, **kwargs) \
            -> Tuple[Output, Optional[HiddenState], torch.LongTensor]:
        r"""Performs decoding. This is a shared interface for both
        :class:`~texar.modules.BasicRNNDecoder` and
        :class:`~texar.modules.AttentionRNNDecoder`.

        Implementation calls initialize() once and step() repeatedly on the
        Decoder object. Please refer to `tf.contrib.seq2seq.dynamic_decode`.

        See Also:
            Arguments of :meth:`create_helper`.

        Args:
            inputs (optional): Input tensors for teacher forcing decoding.
                Used when `decoding_strategy` is set to "train_greedy", or
                when `hparams`-configured helper is used.

                - If :attr:`embedding` is `None`, `inputs` is directly \
                fed to the decoder. E.g., in `"train_greedy"` strategy, \
                `inputs` must be a 3D Tensor of shape \
                `[batch_size, max_time, emb_dim]` (or \
                `[max_time, batch_size, emb_dim]` if `input_time_major`==True).
                - If `embedding` is given, `inputs` is used as index \
                to look up embeddings and feed in the decoder. \
                E.g., if `embedding` is an instance of \
                :class:`~texar.modules.WordEmbedder`, \
                then :attr:`inputs` is usually a 2D int Tensor \
                `[batch_size, max_time]` (or \
                `[max_time, batch_size]` if `input_time_major`==True) \
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
                :tf_main:`Helper <contrib/seq2seq/Helper>`
                that defines the decoding strategy. If given,
                `decoding_strategy`
                and helper configs in :attr:`hparams` are ignored.
            infer_mode (optional): If not `None`, overrides mode given by
                `self.training`.
            **kwargs: Other keyword arguments for constructing helpers
                defined by `hparams["helper_trainn"]` or
                `hparams["helper_infer"]`.

        Returns:
            `(outputs, final_state, sequence_lengths)`, where

            - **`outputs`**: an object containing the decoder output on all \
            time steps.
            - **`final_state`**: is the cell state of the final time step.
            - **`sequence_lengths`**: is an int Tensor of shape `[batch_size]` \
            containing the length of each sample.
        """
        # TODO: Add faster code path for teacher-forcing training.

        # Helper
        if helper is None:
            # Prefer creating a new helper when at least one kwarg is specified.
            prefer_new = (len(kwargs) > 0)
            kwargs.update(infer_mode=infer_mode)
            is_training = (not infer_mode if infer_mode is not None
                           else self.training)
            helper = self._train_helper if is_training else self._infer_helper
            if prefer_new or helper is None:
                helper = self.create_helper(**kwargs)
                if is_training and self._train_helper is None:
                    self._train_helper = helper
                elif not is_training and self._infer_helper is None:
                    self._infer_helper = helper

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
                   initial_state: Optional[HiddenState]) \
            -> Tuple[torch.ByteTensor, torch.Tensor, HiddenState]:
        initial_finished, initial_inputs = helper.initialize(
            inputs, sequence_length)
        state = initial_state or self._cell.init_batch(initial_inputs.size(0))
        return (initial_finished, initial_inputs, state)

    def step(self, helper: Helper, time: int,
             inputs: torch.Tensor, state: Optional[HiddenState]) \
            -> Tuple[Output, HiddenState, torch.Tensor, torch.ByteTensor]:
        raise NotImplementedError

    @property
    def cell(self):
        r"""The RNN cell."""
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
        r"""The output layer."""
        return self._output_layer
