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

from typing import Callable, NamedTuple, Optional, Tuple, TypeVar, Union

import torch
from torch import nn

from texar.core import RNNCellBase
from texar.core.attention_mechanism import *
from texar.core.cell_wrappers import AttentionWrapper
from texar.core.cell_wrappers import HiddenState
from texar.hyperparams import HParams
from texar.modules.decoders.decoder_helpers import Helper
from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.utils.types import MaybeTuple
from texar.utils.utils import check_or_get_instance, get_function

__all__ = [
    'BasicRNNDecoderOutput',
    'AttentionRNNDecoderOutput',
    'BasicRNNDecoder',
    'AttentionRNNDecoder',
]

State = TypeVar('State')


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
             inputs: torch.Tensor, state: Optional[HiddenState]) \
            -> Tuple[BasicRNNDecoderOutput, HiddenState,
                     torch.Tensor, torch.ByteTensor]:
        cell_outputs, cell_state = self._cell(inputs, state)
        logits = self._output_layer(cell_outputs)
        sample_ids = helper.sample(time=time, outputs=logits)
        (finished, next_inputs) = helper.next_inputs(
            time=time,
            outputs=logits,
            sample_ids=sample_ids)
        next_state = cell_state
        outputs = BasicRNNDecoderOutput(logits, sample_ids, cell_outputs)
        return (outputs, next_state, next_inputs, finished)

    @property
    def output_size(self):
        r"""Output size of one step.
        """
        return self._cell.hidden_size


class AttentionRNNDecoder(RNNDecoderBase[AttentionRNNDecoderOutput]):
    """RNN decoder with attention mechanism.

    Args:
        memory: The memory to query, e.g., the output of an RNN encoder. This
            tensor should be shaped `[batch_size, max_time, dim]`.
        memory_sequence_length (optional): A tensor of shape `[batch_size]`
            containing the sequence lengths for the batch
            entries in memory. If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
        cell (RNNCell, optional): An instance of `RNNCell`. If `None`, a cell
            is created as specified in :attr:`hparams`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in
            TRAIN mode). If `None`, :func:`~texar.global_mode` is used.
            Ignored if :attr:`cell` is given.
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An output layer that transforms cell output
            to logits. This can be:

            - A callable layer, e.g., an instance \
            of :tf_main:`tf.layers.Layer <layers/Layer>`.
            - A tensor. A dense layer will be created using the tensor \
            as the kernel weights. The bias of the dense layer is determined by\
            `hparams.output_layer_bias`. This can be used to tie the output \
            layer with the input embedding matrix, as proposed in \
            https://arxiv.org/pdf/1608.05859.pdf
            - `None`. A dense layer will be created based on attr:`vocab_size`\
            and `hparams.output_layer_bias`.
            - If no output layer after the cell output is needed, set \
            `(vocab_size=None, output_layer=tf.identity)`.
        cell_input_fn (callable, optional): A callable that produces RNN cell
            inputs. If `None` (default), the default is used:
            `lambda inputs, attention: tf.concat([inputs, attention], -1)`,
            which cancats regular RNN cell inputs with attentions.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`~texar.modules.RNNDecoderBase._build` for the inputs and outputs
    of the decoder. The decoder returns
    `(outputs, final_state, sequence_lengths)`, where `outputs` is an instance
    of :class:`~texar.modules.AttentionRNNDecoderOutput`.

    Example:

        .. code-block:: python

            # Encodes the source
            enc_embedder = WordEmbedder(data.source_vocab.size, ...)
            encoder = UnidirectionalRNNEncoder(...)

            enc_outputs, _ = encoder(
                inputs=enc_embedder(data_batch['source_text_ids']),
                sequence_length=data_batch['source_length'])

            # Decodes while attending to the source
            dec_embedder = WordEmbedder(vocab_size=data.target_vocab.size, ...)
            decoder = AttentionRNNDecoder(
                memory=enc_outputs,
                memory_sequence_length=data_batch['source_length'],
                vocab_size=data.target_vocab.size)

            outputs, _, _ = decoder(
                decoding_strategy='train_greedy',
                inputs=dec_embedder(data_batch['target_text_ids']),
                sequence_length=data_batch['target_length']-1)
    """

    def __init__(self,
                 memory: torch.Tensor,
                 input_size: int,
                 vocab_size: int,
                 memory_sequence_length: Optional[torch.Tensor] = None,
                 cell: Optional[RNNCellBase] = None,
                 output_layer: Optional[Union[nn.Module, torch.Tensor]] = None,
                 # attention_layer=None, # TODO(zhiting): only valid for tf>=1.0
                 cell_input_fn: Optional[Callable[[torch.Tensor],
                                         torch.Tensor]] = None,
                 hparams: Optional[HParams] = None):

        super().__init__(cell=cell,
                         input_size=input_size,
                         vocab_size=vocab_size,
                         output_layer=output_layer,
                         hparams=hparams)

        self.dtype: torch.dtype = memory.dtype

        attn_hparams = self._hparams['attention']
        attn_kwargs = attn_hparams['kwargs'].todict()

        # Parse the 'probability_fn' argument
        if 'probability_fn' in attn_kwargs:
            prob_fn = attn_kwargs['probability_fn']
            if prob_fn is not None and not callable(prob_fn):
                prob_fn = get_function(prob_fn, ['torch.nn.functional'])
            attn_kwargs['probability_fn'] = prob_fn

        if attn_hparams['type'] in ['BahdanauAttention',
                                    'BahdanauMonotonicAttention']:
            attn_kwargs.update({"cell_output_size": self._cell.hidden_size})

        attn_kwargs.update({
            "memory_sequence_length": memory_sequence_length,
            "memory": memory})
        self._attn_kwargs = attn_kwargs
        attn_modules = ['texar.core']

        self.attention_mechanism: AttentionMechanism
        self.attention_mechanism = check_or_get_instance(
            attn_hparams["type"], attn_kwargs, attn_modules,
            classtype=AttentionMechanism)

        self._attn_cell_kwargs = {
            "attention_layer_size": attn_hparams["attention_layer_size"],
            "alignment_history": attn_hparams["alignment_history"],
            "output_attention": attn_hparams["output_attention"],
        }
        self._cell_input_fn = cell_input_fn

        if attn_hparams["output_attention"] and vocab_size is not None and \
                self.attention_mechanism is not None:
            self._output_layer = nn.Linear(
                self.attention_mechanism.values.shape[-1],
                vocab_size)

        attn_cell = AttentionWrapper(  # type: ignore
            self._cell,
            self.attention_mechanism,
            cell_input_fn=self._cell_input_fn,
            #attention_layer=attention_layer,
            **self._attn_cell_kwargs)

        self._cell = attn_cell
        self._cell: AttentionWrapper

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        Common hyperparameters are the same as in
        :class:`~texar.modules.BasicRNNDecoder`.
        :meth:`~texar.modules.BasicRNNDecoder.default_hparams`.
        Additional hyperparameters are for attention mechanism
        configuration.

        .. code-block:: python

            {
                "attention": {
                    "type": "LuongAttention",
                    "kwargs": {
                        "num_units": 256,
                    },
                    "attention_layer_size": None,
                    "alignment_history": False,
                    "output_attention": True,
                },
                # The following hyperparameters are the same as with
                # `BasicRNNDecoder`
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
                "name": "attention_rnn_decoder"
            }

        Here:

        "attention" : dict
            Attention hyperparameters, including:

            "type" : str or class or instance
                The attention type. Can be an attention class, its name or
                module path, or a class instance. The class must be a subclass
                of :tf_main:`TF AttentionMechanism
                <contrib/seq2seq/AttentionMechanism>`. If class name is
                given, the class must be from modules
                :tf_main:`tf.contrib.seq2seq <contrib/seq2seq>` or
                :mod:`texar.custom`.

                Example:

                    .. code-block:: python

                        # class name
                        "type": "LuongAttention"
                        "type": "BahdanauAttention"
                        # module path
                        "type": "tf.contrib.seq2seq.BahdanauMonotonicAttention"
                        "type": "my_module.MyAttentionMechanismClass"
                        # class
                        "type": tf.contrib.seq2seq.LuongMonotonicAttention
                        # instance
                        "type": LuongAttention(...)

            "kwargs" : dict
                keyword arguments for the attention class constructor.
                Arguments :attr:`memory` and
                :attr:`memory_sequence_length` should **not** be
                specified here because they are given to the decoder
                constructor. Ignored if "type" is an attention class
                instance. For example

                Example:

                    .. code-block:: python

                        "type": "LuongAttention",
                        "kwargs": {
                            "num_units": 256,
                            "probability_fn": tf.nn.softmax
                        }

                    Here "probability_fn" can also be set to the string name
                    or module path to a probability function.

                "attention_layer_size" : int or None
                    The depth of the attention (output) layer. The context and
                    cell output are fed into the attention layer to generate
                    attention at each time step.
                    If `None` (default), use the context as attention at each
                    time step.

                "alignment_history": bool
                    whether to store alignment history from all time steps
                    in the final output state. (Stored as a time major
                    `TensorArray` on which you must call `stack()`.)

                "output_attention": bool
                    If `True` (default), the output at each time step is
                    the attention value. This is the behavior of Luong-style
                    attention mechanisms. If `False`, the output at each
                    time step is the output of `cell`.  This is the
                    beahvior of Bhadanau-style attention mechanisms.
                    In both cases, the `attention` tensor is propagated to
                    the next time step via the state and is used there.
                    This flag only controls whether the attention mechanism
                    is propagated up to the next cell in an RNN stack or to
                    the top RNN output.
        """
        hparams = RNNDecoderBase.default_hparams()
        hparams["name"] = "attention_rnn_decoder"
        hparams["attention"] = {
            "type": "LuongAttention",
            "kwargs": {
                "num_units": 256,
            },
            "attention_layer_size": None,
            "alignment_history": False,
            "output_attention": True,
        }
        return hparams

    def initialize(self,  # type: ignore
                   helper: Helper,
                   inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor],
                   initial_state: Optional[AttentionWrapperState]) -> \
            Tuple[torch.ByteTensor, torch.Tensor,
                  Optional[AttentionWrapperState]]:
        initial_finished, initial_inputs = helper.initialize(
            inputs, sequence_length)

        if inputs is not None:
            batch_size = inputs.shape[0]
        else:
            batch_size = helper._batch_size  # type: ignore

        initial_state = self._cell.zero_state(batch_size=batch_size)
        return initial_finished, initial_inputs, initial_state

    def step(self,  # type: ignore
             helper: Helper,
             time: int,
             inputs: torch.Tensor,
             state: AttentionWrapperState) -> \
            Tuple[AttentionRNNDecoderOutput, AttentionWrapperState,
                  torch.Tensor, torch.ByteTensor]:
        wrapper_outputs, wrapper_state = self._cell(inputs, state)
        # Essentisally the same as in BasicRNNDecoder.step()
        logits = self._output_layer(wrapper_outputs)
        sample_ids = helper.sample(time=time, outputs=logits)
        finished, next_inputs = helper.next_inputs(
            time=time,
            outputs=logits,
            sample_ids=sample_ids)

        attention_scores = wrapper_state.alignments
        attention_context = wrapper_state.attention
        outputs = AttentionRNNDecoderOutput(
            logits, sample_ids, wrapper_outputs,
            attention_scores, attention_context)
        next_state = wrapper_state

        return outputs, next_state, next_inputs, finished

    @property
    def output_size(self):
        r"""Output size of one step.
                """
        return self._cell.hidden_size

    def wrapper_zero_state(self,
                           batch_size: int) -> AttentionWrapperState:
        """Returns zero state of the attention-wrapped cell.
        Equivalent to :attr:`decoder.cell.zero_state`.
        """
        return self._cell.zero_state(batch_size=batch_size)

    @property
    def wrapper_state_size(self) -> AttentionWrapperState:
        """The state size of the attention-wrapped cell.
        Equivalent to :attr:`decoder.cell.state_size`.
        """
        return self._cell.state_size
