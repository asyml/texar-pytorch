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
Various RNN decoders.
"""

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager, protected-access, invalid-name

from typing import NamedTuple, Tuple

import torch

from texar.core.cell_wrappers import HiddenState
from texar.modules.decoders.decoder_helpers import Helper
from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.utils.types import MaybeTuple

__all__ = [
    'BasicRNNDecoderOutput',
    'AttentionRNNDecoderOutput',
    'BasicRNNDecoder',
]


class BasicRNNDecoderOutput(NamedTuple):
    r"""The outputs of basic RNN decoder that include both RNN outputs and
    sampled ids at each step. This is also used to store results of all the
    steps after decoding the whole sequence.

    Attributes:
        logits: The outputs of RNN (at each step/of all steps) by applying the
            output layer on cell outputs. E.g., in
            :class:`~texar.modules.BasicRNNDecoder` with default
            hyperparameters, this is a Tensor of
            shape `[batch_size, max_time, vocab_size]` after decoding the
            whole sequence.
        sample_id: The sampled results (at each step/of all steps). E.g., in
            BasicRNNDecoder with decoding strategy of train_greedy,
            this is a Tensor
            of shape `[batch_size, max_time]` containing the sampled token
            indexes of all steps.
        cell_output: The output of RNN cell (at each step/of all steps).
            This is the results prior to the output layer. E.g., in
            BasicRNNDecoder with default
            hyperparameters, this is a Tensor of
            shape `[batch_size, max_time, cell_output_size]` after decoding
            the whole sequence.
    """
    logits: torch.Tensor
    sample_id: torch.LongTensor
    cell_output: torch.Tensor


class AttentionRNNDecoderOutput(NamedTuple):
    r"""The outputs of attention RNN decoders that additionally include
    attention results.

    Attributes:
        logits: The outputs of RNN (at each step/of all steps) by applying the
            output layer on cell outputs. E.g., in
            :class:`~texar.modules.AttentionRNNDecoder`, this is a Tensor of
            shape `[batch_size, max_time, vocab_size]` after decoding.
        sample_id: The sampled results (at each step/of all steps). E.g., in
            :class:`~texar.modules.AttentionRNNDecoder` with decoding strategy
            of train_greedy, this
            is a Tensor of shape `[batch_size, max_time]` containing the
            sampled token indexes of all steps.
        cell_output: The output of RNN cell (at each step/of all steps).
            This is the results prior to the output layer. E.g., in
            AttentionRNNDecoder with default
            hyperparameters, this is a Tensor of
            shape `[batch_size, max_time, cell_output_size]` after decoding
            the whole sequence.
        attention_scores: A single or tuple of `Tensor`(s) containing the
            alignments emitted (at the previous time step/of all time steps)
            for each attention mechanism.
        attention_context: The attention emitted (at the previous time step/of
            all time steps).
    """
    logits: torch.Tensor
    sample_id: torch.LongTensor
    cell_output: torch.Tensor
    attention_scores: MaybeTuple[torch.Tensor]
    attention_context: torch.Tensor


class BasicRNNDecoder(RNNDecoderBase[BasicRNNDecoderOutput]):
    r"""Basic RNN decoder.

    Args:
        cell (RNNCell, optional): An instance of
            :tf_main:`RNNCell <ontrib/rnn/RNNCell>`. If `None`
            (default), a cell is created as specified in
            :attr:`hparams`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in
            TRAIN mode). If `None`, :func:`~texar.global_mode` is used.
            Ignored if :attr:`cell` is given.
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`, or
            :tf_main:`tf.identity <identity>`. Apply to the RNN cell
            output to get logits. If `None`, a dense layer
            is used with output dimension set to :attr:`vocab_size`.
            Set `output_layer=tf.identity` if you do not want to have an
            output layer after the RNN cell outputs.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`~texar.modules.RNNDecoderBase._build` for the inputs and outputs
    of the decoder. The decoder returns
    `(outputs, final_state, sequence_lengths)`, where `outputs` is an instance
    of :class:`~texar.modules.BasicRNNDecoderOutput`.

    Example:

        .. code-block:: python

            embedder = WordEmbedder(vocab_size=data.vocab.size)
            decoder = BasicRNNDecoder(vocab_size=data.vocab.size)

            # Training loss
            outputs, _, _ = decoder(
                decoding_strategy='train_greedy',
                inputs=embedder(data_batch['text_ids']),
                sequence_length=data_batch['length']-1)

            loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=data_batch['text_ids'][:, 1:],
                logits=outputs.logits,
                sequence_length=data_batch['length']-1)

            # Inference sample
            outputs, _, _ = decoder(
                decoding_strategy='infer_sample',
                start_tokens=[data.vocab.bos_token_id]*100,
                end_token=data.vocab.eos.token_id,
                embedding=embedder,
                max_decoding_length=60,
                mode=tf.estimator.ModeKeys.PREDICT)

            sample_id = sess.run(outputs.sample_id)
            sample_text = tx.utils.map_ids_to_strs(sample_id, data.vocab)
            print(sample_text)
            # [
            #   the first sequence sample .
            #   the second sequence sample .
            #   ...
            # ]
    """

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "rnn_cell": default_rnn_cell_hparams(),
                "max_decoding_length_train": None,
                "max_decoding_length_infer": None,
                "helper_train": {
                    "type": "TrainingHelper",
                    "kwargs": {}
                }
                "helper_infer": {
                    "type": "SampleEmbeddingHelper",
                    "kwargs": {}
                }
                "name": "basic_rnn_decoder"
            }

        Here:

        "rnn_cell" : dict
            A dictionary of RNN cell hyperparameters. Ignored if
            :attr:`cell` is given to the decoder constructor.
            The default value is defined in
            :func:`~texar.core.default_rnn_cell_hparams`.

        "max_decoding_length_train": int or None
            Maximum allowed number of decoding steps in training mode.
            If `None` (default), decoding is
            performed until fully done, e.g., encountering the <EOS> token.
            Ignored if `max_decoding_length` is given when calling
            the decoder.

        "max_decoding_length_infer" : int or None
            Same as "max_decoding_length_train" but for inference mode.

        "helper_train" : dict
            The hyperparameters of the helper used in training.
            "type" can be a helper class, its name or module path, or a
            helper instance. If a class name is given, the class must be
            from module :tf_main:`tf.contrib.seq2seq <contrib/seq2seq>`,
            :mod:`texar.modules`, or :mod:`texar.custom`. This is used
            only when both `decoding_strategy` and `helper` augments are
            `None` when calling the decoder. See
            :meth:`~texar.modules.RNNDecoderBase._build` for more details.

        "helper_infer": dict
            Same as "helper_train" but during inference mode.

        "name" : str
            Name of the decoder.

            The default value is "basic_rnn_decoder".
        """
        hparams = RNNDecoderBase.default_hparams()
        hparams['name'] = 'basic_rnn_decoder'
        return hparams

    def step(self, helper: Helper, time: int,
             inputs: torch.Tensor, state: HiddenState) \
            -> Tuple[BasicRNNDecoderOutput, HiddenState,
                     torch.Tensor, torch.ByteTensor]:
        cell_outputs, cell_state = self._cell(inputs, state)
        logits = self._output_layer(cell_outputs)
        sample_ids = helper.sample(
            time=time, outputs=logits, state=cell_state)
        (finished, next_inputs, next_state) = helper.next_inputs(
            time=time,
            outputs=logits,
            state=cell_state,
            sample_ids=sample_ids)
        outputs = BasicRNNDecoderOutput(logits, sample_ids, cell_outputs)
        return (outputs, next_state, next_inputs, finished)

    @property
    def output_size(self):
        r"""Output size of one step.
        """
        return BasicRNNDecoderOutput(
            logits=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            cell_output=self._cell.output_size)
