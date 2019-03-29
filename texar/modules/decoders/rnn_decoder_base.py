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

import copy
from typing import Generic, Optional, Tuple, TypeVar

import torch
from torch import nn

from texar import HParams
from texar.core import layers
from texar.core.cell_wrappers import HiddenState
from texar.module_base import ModuleBase
from texar.modules.decoders import rnn_decoder_helpers
from texar.modules.decoders.rnn_decoder_helpers import Helper
from texar.utils import utils

__all__ = [
    'DecoderInitTuple',
    'StepOutputTuple',
    'RNNDecoderBase',
]

DecoderInitTuple = Tuple[torch.ByteTensor, torch.Tensor, HiddenState]

Output = TypeVar('Output')  # output type can be of any nested structure
StepOutputTuple = Tuple[Output, HiddenState, torch.Tensor, torch.ByteTensor]


class RNNDecoderBase(ModuleBase, Generic[Output]):
    r"""Base class inherited by all RNN decoder classes.
    See :class:`~texar.modules.BasicRNNDecoder` for the argumenrts.

    See :meth:`_build` for the inputs and outputs of RNN decoders in general.
    """

    def __init__(self,
                 cell=None,
                 vocab_size: Optional[int] = None,
                 output_layer: Optional[nn.Module] = None,
                 input_time_major: bool = False,
                 output_time_major: bool = False,
                 hparams: Optional[HParams] = None):
        ModuleBase.__init__(self, hparams)

        self._train_helper: Optional[Helper] = None
        self._infer_helper: Optional[Helper] = None
        self._input_time_major = input_time_major
        self._output_time_major = output_time_major

        # Make RNN cell
        self._cell = cell or layers.get_rnn_cell(self._hparams.rnn_cell)
        self._beam_search_cell = None

        # Make the output layer
        self._vocab_size = vocab_size
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
            'helper_train': rnn_decoder_helpers.default_helper_train_hparams(),
            'helper_infer': rnn_decoder_helpers.default_helper_infer_hparams(),
            'max_decoding_length_train': None,
            'max_decoding_length_infer': None,
            'name': 'rnn_decoder',
        }

    def create_helper(self, *,
                      decoding_strategy: Optional[str] = None,
                      embedding: Optional[nn.Module] = None,
                      start_tokens: Optional[torch.LongTensor] = None,
                      end_token: Optional[int] = None,
                      softmax_temperature: Optional[float] = None,
                      infer_mode: Optional[bool] = None,
                      **kwargs) -> Helper:
        r"""Create a helper instance for the decoder. This is a shared interface
        for both :class:`~texar.modules.BasicRNNDecoder` and
        :class:`~texar.modules.AttentionRNNDecoder`.

        The function provides **3 ways** to specify the
        decoding method, with varying flexibility:

        1. The :attr:`decoding_strategy` argument: A string taking value of:

            - **"train_greedy"**: decoding in teacher-forcing fashion (i.e.,
              feeding `ground truth` to decode the next step), and each sample
              is obtained by taking the `argmax` of the RNN output logits.
              Arguments :attr:`(inputs, sequence_length, input_time_major)`
              are required for this strategy, and argument :attr:`embedding`
              is optional.
            - **"infer_greedy"**: decoding in inference fashion (i.e., feeding
              the `generated` sample to decode the next step), and each sample
              is obtained by taking the `argmax` of the RNN output logits.
              Arguments :attr:`(embedding, start_tokens, end_token)` are
              required for this strategy, and argument
              :attr:`max_decoding_length` is optional.
            - **"infer_sample"**: decoding in inference fashion, and each
              sample is obtained by `random sampling` from the RNN output
              distribution. Arguments
              :attr:`(embedding, start_tokens, end_token)` are
              required for this strategy, and argument
              :attr:`max_decoding_length` is optional.

          This argument is used only when argument :attr:`helper` is `None`.

          Example:

            .. code-block:: python

                embedder = WordEmbedder(vocab_size=data.vocab.size)
                decoder = BasicRNNDecoder(vocab_size=data.vocab.size)

                # Teacher-forcing decoding
                outputs_1, _, _ = decoder(
                    decoding_strategy='train_greedy',
                    inputs=embedder(data_batch['text_ids']),
                    sequence_length=data_batch['length']-1)

                # Random sample decoding. Gets 100 sequence samples
                outputs_2, _, sequence_length = decoder(
                    decoding_strategy='infer_sample',
                    start_tokens=[data.vocab.bos_token_id]*100,
                    end_token=data.vocab.eos.token_id,
                    embedding=embedder,
                    max_decoding_length=60)

        2. The :attr:`helper` argument: An instance of subclass of
           :class:`~texar.modules.decoders.rnn_decoder_helpers.Helper`. This
           provides a superset of decoding strategies than above, for example:

            - :class:`~texar.modules.TrainingHelper` corresponding to the
              "train_greedy" strategy.
            - :class:`~texar.modules.ScheduledEmbeddingTrainingHelper` and
              :class:`~texar.modules.ScheduledOutputTrainingHelper` for
              scheduled sampling.
            - :class:`~texar.modules.SoftmaxEmbeddingHelper` and
              :class:`~texar.modules.GumbelSoftmaxEmbeddingHelper` for
              soft decoding and gradient backpropagation.

          This means gives the maximal flexibility of configuring the decoding
          strategy.

          Example:

            .. code-block:: python

                embedder = WordEmbedder(vocab_size=data.vocab.size)
                decoder = BasicRNNDecoder(vocab_size=data.vocab.size)

                # Teacher-forcing decoding, same as above with
                # `decoding_strategy='train_greedy'`
                helper_1 = TrainingHelper(
                    inputs=embedders(data_batch['text_ids']),
                    sequence_length=data_batch['length']-1)
                outputs_1, _, _ = decoder(helper=helper_1)

                # Gumbel-softmax decoding
                helper_2 = GumbelSoftmaxEmbeddingHelper(
                    embedding=embedder,
                    start_tokens=[data.vocab.bos_token_id]*100,
                    end_token=data.vocab.eos_token_id,
                    tau=0.1)
                outputs_2, _, sequence_length = decoder(
                    max_decoding_length=60, helper=helper_2)

        3. :attr:`hparams["helper_train"]` and :attr:`hparams["helper_infer"]`:
           Specifying the helper through hyperparameters. Train and infer
           strategy is toggled based on :attr:`mode`. Appriopriate arguments
           (e.g., :attr:`inputs`, :attr:`start_tokens`, etc) are selected to
           construct the helper. Additional arguments for helper constructor
           can be provided either through :attr:`**kwargs`, or through
           :attr:`hparams["helper_train/infer"]["kwargs"]`.

           This means is used only when both :attr:`decoding_strategy` and
           :attr:`helper` are `None`.

           Example:

             .. code-block:: python

                h = {
                    "helper_infer": {
                        "type": "GumbelSoftmaxEmbeddingHelper",
                        "kwargs": { "tau": 0.1 }
                    }
                }
                embedder = WordEmbedder(vocab_size=data.vocab.size)
                decoder = BasicRNNDecoder(vocab_size=data.vocab.size, hparams=h)

                # Gumbel-softmax decoding
                output, _, _ = decoder(
                    decoding_strategy=None, # Sets to None explicit
                    embedding=embedder,
                    start_tokens=[data.vocab.bos_token_id]*100,
                    end_token=data.vocab.eos_token_id,
                    max_decoding_length=60,
                    mode=tf.estimator.ModeKeys.PREDICT)
                        # PREDICT mode also shuts down dropout

        Args:
            decoding_strategy (str): A string specifying the decoding
                strategy. Different arguments are required based on the
                strategy.
                Ignored if :attr:`helper` is given.
            embedding (optional): A callable that returns embedding vectors
                of `inputs` (e.g., an instance of subclass of
                :class:`~texar.modules.EmbedderBase`), or the `params`
                argument of
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`.
                If provided, `inputs` (if used) will be passed to
                `embedding` to fetch the embedding vectors of the inputs.
                Required when `decoding_strategy="infer_greedy"`
                or `"infer_sample"`; optional when
                `decoding_strategy="train_greedy"`.
            start_tokens (optional): A int Tensor of shape `[batch_size]`,
                the start tokens.
                Used when `decoding_strategy="infer_greedy"` or
                `"infer_sample"`, or when `hparams`-configured
                helper is used.
                Companying with Texar data module, to get `batch_size` samples
                where batch_size is changing according to the data module,
                this can be set as
                `start_tokens=tf.ones_like(batch['length'])*bos_token_id`.
            end_token (optional): A int 0D Tensor, the token that marks end
                of decoding.
                Used when `decoding_strategy="infer_greedy"` or
                `"infer_sample"`, or when `hparams`-configured
                helper is used.
            softmax_temperature (optional): A float 0D Tensor, value to divide
                the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples. Must > 0. If `None`,
                1.0 is used.
                Used when `decoding_strategy="infer_sample"`.
            infer_mode (optional): If not `None`, overrides mode given by
                `self.training`.
            **kwargs: Other keyword arguments for constructing helpers
                defined by `hparams["helper_trainn"]` or
                `hparams["helper_infer"]`.

        Returns:
            The constructed helper instance.
        """
        if decoding_strategy is not None:
            if decoding_strategy == 'train_greedy':
                helper: Helper = rnn_decoder_helpers.TrainingHelper(
                    embedding, self._input_time_major)
            elif decoding_strategy in ['infer_greedy', 'infer_sample']:
                if (embedding is None or
                        start_tokens is None or
                        end_token is None):
                    raise ValueError(
                        f"When using '{decoding_strategy}' decoding strategy, "
                        f"'embedding', 'start_tokens', and 'end_token' must not"
                        f"be `None`.")
                if decoding_strategy == 'infer_greedy':
                    helper = rnn_decoder_helpers.GreedyEmbeddingHelper(
                        embedding, start_tokens, end_token)
                else:
                    helper = rnn_decoder_helpers.SampleEmbeddingHelper(
                        embedding, start_tokens, end_token, softmax_temperature)
            else:
                raise ValueError(
                    f"Unknown decoding strategy: {decoding_strategy}")
        else:
            is_training = (not infer_mode if infer_mode is not None
                           else self.training)
            if is_training:
                kwargs_ = copy.copy(self._hparams.helper_train.kwargs.todict())
                helper_type = self._hparams.helper_train.type
            else:
                kwargs_ = copy.copy(self._hparams.helper_infer.kwargs.todict())
                helper_type = self._hparams.helper_infer.type
            kwargs_.update({
                'time_major': self._input_time_major,
                'embedding': embedding,
                'start_tokens': start_tokens,
                'end_token': end_token,
                'softmax_temperature': softmax_temperature})
            kwargs_.update(kwargs)
            helper = rnn_decoder_helpers.get_helper(helper_type, **kwargs_)
        return helper

    def set_default_train_helper(self, helper: Helper):
        r"""Set the default helper used in training mode.

        Args:
            helper: the helper to set as default training helper.
        """
        self._train_helper = helper

    def set_default_infer_helper(self, helper: Helper):
        r"""Set the default helper used in eval (inference) mode.

        Args:
            helper: the helper to set as default inference helper.
        """
        self._infer_helper = helper

    @classmethod
    def _get_batch_size_from_state(cls, state: HiddenState) -> int:
        if isinstance(state, (list, tuple)):
            return cls._get_batch_size_from_state(state[0])
        return state.size(0)

    def forward(self, inputs: Optional[torch.Tensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                initial_state: Optional[HiddenState] = None,
                helper: Optional[Helper] = None,
                max_decoding_length: Optional[int] = None,
                impute_finished: bool = False,
                output_time_major: bool = False,
                infer_mode: Optional[bool] = None,
                **kwargs):
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
            output_time_major (bool): If `True`, outputs are returned as
                time major tensors. If `False` (default), outputs are returned
                as batch major tensors.
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

        if (isinstance(helper, rnn_decoder_helpers.TrainingHelper) and
                (inputs is None or sequence_length is None)):
            raise ValueError("'input' and 'sequence_length' must not be None "
                             "when using 'TrainingHelper'.")

        # Initial state
        if initial_state is not None:
            batch_size = self._get_batch_size_from_state(initial_state)
            self._cell.init_batch(batch_size)

        # Maximum decoding length
        if max_decoding_length is None:
            if self.training:
                max_decoding_length = self._hparams.max_decoding_length_train
            else:
                max_decoding_length = self._hparams.max_decoding_length_infer
            if max_decoding_length is None:
                max_decoding_length = utils.MAX_SEQ_LENGTH

        # Decode
        finished, step_inputs, state = self.initialize(  # type: ignore
            helper, inputs, sequence_length, initial_state)

        zero_outputs = step_inputs.new_zeros(
            step_inputs.size(0), self._cell.hidden_size)

        if max_decoding_length is not None:
            finished |= (max_decoding_length <= 0)
        sequence_lengths = torch.zeros_like(
            finished, dtype=torch.long, device=finished.device)
        time = 0

        outputs = []

        while (not torch.all(finished).item() and
               (max_decoding_length is None or time < max_decoding_length)):
            (next_outputs, decoder_state, next_inputs,
             decoder_finished) = self.step(helper, time, step_inputs, state)

            if getattr(self, 'tracks_own_finished', False):
                next_finished = decoder_finished
            else:
                next_finished = decoder_finished | finished

            # Zero out output values past finish
            if impute_finished:
                emit = utils.map_structure_zip(
                    lambda new, cur: torch.where(finished, cur, new),
                    (next_outputs, zero_outputs))
                next_state = utils.map_structure_zip(
                    lambda new, cur: torch.where(finished, cur, new),
                    (decoder_state, state))
            else:
                emit = next_outputs
                next_state = decoder_state

            outputs.append(emit)
            sequence_lengths.index_fill_(
                dim=0, value=time + 1,
                index=torch.nonzero((~finished).long()).flatten())
            time += 1
            finished = next_finished
            step_inputs = next_inputs
            state = next_state

        final_outputs = utils.map_structure_zip(
            lambda *tensors: torch.stack(tensors),
            outputs)  # output at each time step may be a namedtuple
        final_state = state
        final_sequence_lengths = sequence_lengths

        try:
            final_outputs, final_state = self.finalize(
                final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass

        if not output_time_major:
            final_outputs = utils.map_structure(
                lambda x: x.transpose(0, 1) if x.dim() >= 2 else x,
                final_outputs)

        return final_outputs, final_state, final_sequence_lengths

    def _get_beam_search_cell(self):
        self._beam_search_cell = self._cell
        return self._cell

    @property
    def output_size(self):
        r"""Output size of one step.
        """
        raise NotImplementedError

    def initialize(self, helper: Helper,
                   inputs: torch.Tensor,
                   sequence_length: torch.LongTensor,
                   initial_state: HiddenState) -> DecoderInitTuple:
        r"""Called before any decoding iterations.

        This methods must compute initial input values and initial state.

        Args:
            helper: The :class:`~texar.modules.Helper` instance to use.
            inputs: A (structure of) input tensors.
            sequence_length: An int32 vector tensor.
            initial_state: Initial RNNCell state (possibly nested tuple of)
                tensor[s].

        Returns:
            `(finished, initial_inputs, initial_state)`: initial values of
            'finished' flags, inputs and state.
        """
        raise NotImplementedError

    def step(self, helper: Helper, time: int, inputs: torch.Tensor,
             state: HiddenState) -> StepOutputTuple[Output]:
        r"""Called per step of decoding (but only once for dynamic decoding).

        Args:
            helper: The :class:`~texar.modules.Helper` instance to use.
            time: Scalar `int32` tensor. Current step number.
            inputs: RNNCell input (possibly nested tuple of) tensor[s] for this
                time step.
            state: RNNCell state (possibly nested tuple of) tensor[s] from
                previous time step.

        Returns:
            `(outputs, next_state, next_inputs, finished)`: `outputs` is an
            object containing the decoder output, `next_state` is a (structure
            of) state tensors and TensorArrays, `next_inputs` is the tensor that
            should be used as input for the next step, `finished` is a boolean
            tensor telling whether the sequence is complete, for each sequence
            in the batch.
        """
        raise NotImplementedError

    def finalize(self, outputs: Output, final_state: HiddenState,
                 sequence_lengths: torch.LongTensor) \
            -> Tuple[Output, HiddenState]:
        r"""Called after all decoding iterations have finished.

        Args:
            outputs: Outputs at each time step.
            final_state: The RNNCell state after the last time step.
            sequence_lengths: Sequence lengths for each sequence in batch.

        Returns:
            `(outputs, final_state)`: `outputs` is an object containing the
            decoder output, `final_state` is a (structure of) state tensors
            and TensorArrays.
        """
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
    def vocab_size(self):
        r"""The vocab size."""
        return self._vocab_size

    @property
    def output_layer(self):
        r"""The output layer."""
        return self._output_layer
