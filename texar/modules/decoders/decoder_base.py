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
Base class for decoders.
"""

import copy
from abc import ABC
from typing import Generic, Optional, Tuple, TypeVar, Union, overload, Callable

import torch
from torch import nn

from texar.core.layers import identity
from texar.module_base import ModuleBase
from texar.modules.decoders import decoder_helpers
from texar.modules.decoders.decoder_helpers import Embedding, Helper
from texar.utils import utils

__all__ = [
    '_make_output_layer',
    'DecoderBase',
]

State = TypeVar('State')
Output = TypeVar('Output')  # output type can be of any nested structure


def _make_output_layer(layer: Optional[Union[nn.Module, torch.Tensor]],
                       vocab_size: Optional[int],
                       output_size: int,
                       bias: bool) -> Tuple[nn.Module, Optional[int]]:
    r"""Construct the output layer for decoders. Based on the input, multiple
    types of output layers could be constructed:

    - If ``layer`` is a :torch_nn:`Module`, then the layer is returned as is.
    - If ``layer`` is `None`, then a :torch_nn:`Linear` layer is constructed
      with ``output_size`` and ``vocab_size`` as input and output dimensions.
    - If ``layer`` is a :tensor:`Tensor`, then a :torch_nn:`Linear` layer is
      constructed with the provided tensor as parameters. Note that this tensor
      should have transposed shape, i.e. shape of ``[vocab_size, output_size]``.
      Also, if the provided tensor is not an instance of :torch_nn:`Parameter`,
      it will **not** accumulate gradients.
    - If ``layer`` is :method:`texar.core.identity`, identity function is used
      as the output layer.
    """
    if isinstance(layer, nn.Module):
        output_layer = layer
    elif layer is None:
        if vocab_size is None:
            raise ValueError(
                "Either `output_layer` or `vocab_size` must be provided. "
                "Set `output_layer=texar.core.identity` if no output layer is "
                "wanted.")
        output_layer = nn.Linear(output_size, vocab_size, bias)
    elif torch.is_tensor(layer):
        vocab_size = layer.size(0)
        output_layer = nn.Linear(layer.size(1), vocab_size, bias)
        if not isinstance(layer, nn.Parameter):
            layer = nn.Parameter(layer, requires_grad=False)
        output_layer.weight = layer
    elif layer is identity:
        output_layer = identity  # type: ignore
    else:
        raise ValueError(
            f"output_layer should be an instance of `nn.Module`, a tensor,"
            f"or None. Unsupported type: {type(layer)}")

    return output_layer, vocab_size


class DecoderBase(ModuleBase, Generic[State, Output], ABC):
    r"""Base class inherited by all RNN decoder classes.
    See :class:`~texar.modules.BasicRNNDecoder` for the arguments.

    See :meth:`forward` for the inputs and outputs of RNN decoders in general.
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: Optional[int] = None,
                 input_time_major: bool = False,
                 output_time_major: bool = False,
                 hparams=None):
        super().__init__(hparams)

        self._train_helper: Optional[Helper] = None
        self._infer_helper: Optional[Helper] = None
        self._input_time_major = input_time_major
        self._output_time_major = output_time_major

        self._input_size = input_size
        self._vocab_size = vocab_size

    def create_helper(self, *,
                      decoding_strategy: Optional[str] = None,
                      embedding: Optional[Embedding] = None,
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
              is obtained by taking the `argmax` of the output logits.
              Arguments :attr:`(inputs, sequence_length)`
              are required for this strategy, and argument :attr:`embedding`
              is optional.
            - **"infer_greedy"**: decoding in inference fashion (i.e., feeding
              the `generated` sample to decode the next step), and each sample
              is obtained by taking the `argmax` of the output logits.
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
                    sequence_length=data_batch['length'] - 1)

                # Random sample decoding. Gets 100 sequence samples
                outputs_2, _, sequence_length = decoder(
                    decoding_strategy='infer_sample',
                    start_tokens=[data.vocab.bos_token_id] * 100,
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
                    sequence_length=data_batch['length'] - 1)
                outputs_1, _, _ = decoder(helper=helper_1)

                # Gumbel-softmax decoding
                helper_2 = GumbelSoftmaxEmbeddingHelper(
                    embedding=embedder,
                    start_tokens=[data.vocab.bos_token_id] * 100,
                    end_token=data.vocab.eos_token_id,
                    tau=0.1)
                outputs_2, _, sequence_length = decoder(
                    max_decoding_length=60, helper=helper_2)

        3. ``hparams["helper_train"]`` and ``hparams["helper_infer"]``:
           Specifying the helper through hyperparameters. Train and infer
           strategy is toggled based on :attr:`mode`. Appropriate arguments
           (e.g., :attr:`inputs`, :attr:`start_tokens`, etc) are selected to
           construct the helper. Additional arguments for helper constructor
           can be provided either through :attr:`**kwargs`, or through
           ``hparams["helper_train/infer"]["kwargs"]``.

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
                decoder.eval()  # disable dropout
                output, _, _ = decoder(
                    decoding_strategy=None, # Sets to None explicit
                    embedding=embedder,
                    start_tokens=[data.vocab.bos_token_id] * 100,
                    end_token=data.vocab.eos_token_id,
                    max_decoding_length=60)

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
                Required when :attr:`decoding_strategy` is ``"infer_greedy"``
                or ``"infer_sample"``; optional when
                ``decoding_strategy="train_greedy"``.
            start_tokens (optional): A :tensor:`LongTensor` of shape
                ``[batch_size]``, the start tokens.
                Used when :attr:`decoding_strategy` is ``"infer_greedy"`` or
                ``"infer_sample"``, or when `hparams`-configured
                helper is used.
                When used with the Texar data module, to get ``batch_size``
                samples where ``batch_size`` is changing according to the data
                module, this can be set as
                ``start_tokens=torch.full_like(batch['length'], bos_token_id)``.
            end_token (optional): A integer or 0D :tensor:`LongTensor`, the
                token that marks the end of decoding.
                Used when :attr:`decoding_strategy` is ``"infer_greedy"`` or
                ``"infer_sample"``, or when `hparams`-configured helper is used.
            softmax_temperature (float, optional): Value to divide the logits
                by before computing the softmax. Larger values (above 1.0)
                result in more random samples. Must be > 0. If `None`, 1.0 is
                used. Used when ``decoding_strategy="infer_sample"``.
            infer_mode (optional): If not `None`, overrides mode given by
                :attr:`self.training`.
            **kwargs: Other keyword arguments for constructing helpers
                defined by ``hparams["helper_train"]`` or
                ``hparams["helper_infer"]``.

        Returns:
            The constructed helper instance.
        """
        if decoding_strategy is not None:
            if decoding_strategy == 'train_greedy':
                helper: Helper = decoder_helpers.TrainingHelper(
                    embedding, self._input_time_major)
            elif decoding_strategy in ['infer_greedy', 'infer_sample']:
                if (embedding is None or
                        start_tokens is None or
                        end_token is None):
                    raise ValueError(
                        f"When using '{decoding_strategy}' decoding strategy, "
                        f"'embedding', 'start_tokens', and 'end_token' must "
                        f"not be `None`.")
                if decoding_strategy == 'infer_greedy':
                    helper = decoder_helpers.GreedyEmbeddingHelper(
                        embedding, start_tokens, end_token)
                else:
                    helper = decoder_helpers.SampleEmbeddingHelper(
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
            helper = decoder_helpers.get_helper(helper_type, **kwargs_)
        return helper

    def _create_or_get_helper(self, infer_mode: Optional[bool] = None,
                              **kwargs) -> Helper:
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
        return helper

    def set_default_train_helper(self, helper: Helper):
        r"""Set the default helper used in training mode.

        Args:
            helper: The helper to set as default training helper.
        """
        self._train_helper = helper

    def set_default_infer_helper(self, helper: Helper):
        r"""Set the default helper used in eval (inference) mode.

        Args:
            helper: The helper to set as default inference helper.
        """
        self._infer_helper = helper

    def dynamic_decode(self, helper: Helper, inputs: Optional[torch.Tensor],
                       sequence_length: Optional[torch.LongTensor],
                       initial_state: Optional[State],
                       max_decoding_length: Optional[int] = None,
                       impute_finished: bool = False,
                       step_hook: Optional[Callable[[int], None]] = None) \
            -> Tuple[Output, Optional[State], torch.LongTensor]:
        r"""Generic routine for dynamic decoding. Please check the
        `documentation
        <https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode>`_
        for the TensorFlow counterpart.

        Returns:
            A tuple of output, final state, and sequence lengths. Note that
            final state could be `None`, when all sequences are of zero length
            and :attr:`initial_state` is also `None`.
        """

        # Decode
        finished, step_inputs, state = self.initialize(
            helper, inputs, sequence_length, initial_state)

        zero_outputs = step_inputs.new_zeros(
            step_inputs.size(0), self.output_size)

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

            if step_hook is not None:
                step_hook(time)

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

        if not self._output_time_major:
            final_outputs = utils.map_structure(
                lambda x: x.transpose(0, 1) if x.dim() >= 2 else x,
                final_outputs)

        return final_outputs, final_state, final_sequence_lengths

    @property
    def output_size(self):
        r"""Output size of one step.
        """
        raise NotImplementedError

    def initialize(self, helper: Helper, inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor],
                   initial_state: Optional[State]) \
            -> Tuple[torch.ByteTensor, torch.Tensor, Optional[State]]:
        r"""Called before any decoding iterations.

        This methods must compute initial input values and initial state.

        Args:
            helper: The :class:`~texar.modules.Helper` instance to use.
            inputs (optional): A (structure of) input tensors.
            sequence_length (optional): A :tensor:`LongTensor` representing
                lengths of each sequence.
            initial_state: A possibly nested structure of tensors indicating the
                initial decoder state.

        Returns:
            A tuple ``(finished, initial_inputs, initial_state)`` representing
            initial values of ``finished`` flags, inputs, and state.
        """
        raise NotImplementedError

    def step(self, helper: Helper, time: int,
             inputs: torch.Tensor, state: Optional[State]) \
            -> Tuple[Output, State, torch.Tensor, torch.ByteTensor]:
        r"""Called per step of decoding (but only once for dynamic decoding).

        Args:
            helper: The :class:`~texar.modules.Helper` instance to use.
            time (int): Current step number.
            inputs: Inputs for this time step.
            state: Decoder state from the previous time step.

        Returns:
            A tuple ``(outputs, next_state, next_inputs, finished)``.

            - ``outputs`` is an object containing the decoder output.
            - ``next_state`` is the decoder state for the next time step.
            - ``next_inputs`` is the tensor that should be used as input for the
              next step.
            - ``finished`` is a :torch:`ByteTensor` tensor telling whether the
              sequence is complete, for each sequence in the batch.
        """
        raise NotImplementedError

    # TODO: Remove these once pylint supports function stubs.
    # pylint: disable=missing-docstring,unused-argument,no-self-use
    # pylint: disable=function-redefined

    @overload
    def finalize(self, outputs: Output, final_state: State,
                 sequence_lengths: torch.LongTensor) -> Tuple[Output, State]:
        ...

    @overload
    def finalize(self, outputs: Output, final_state: Optional[State],
                 sequence_lengths: torch.LongTensor) \
            -> Tuple[Output, Optional[State]]:
        ...

    def finalize(self, outputs, final_state, sequence_lengths):
        r"""Called after all decoding iterations have finished.

        Args:
            outputs: Outputs at each time step.
            final_state: The RNNCell state after the last time step.
            sequence_lengths: Sequence lengths for each sequence in batch.

        Returns:
            A tuple ``(outputs, final_state)``.

            - ``outputs`` is an object containing the decoder output.
            - ``final_state`` is the final decoder state.
        """
        return outputs, final_state

    # pylint: enable=missing-docstring,unused-argument,no-self-use
    # pylint: enable=function-redefined

    @property
    def vocab_size(self):
        r"""The vocabulary size.
        """
        return self._vocab_size

    @property
    def output_layer(self):
        r"""The output layer.
        """
        return self._output_layer
