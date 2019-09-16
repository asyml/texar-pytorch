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

from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn

from texar.torch.core import layers
from texar.torch.core.attention_mechanism import (
    AttentionMechanism, AttentionWrapperState)
from texar.torch.core.cell_wrappers import (
    AttentionWrapper, HiddenState, RNNCellBase)
from texar.torch.modules.decoders import decoder_helpers
from texar.torch.modules.decoders.decoder_base import (
    TokenEmbedder, TokenPosEmbedder)
from texar.torch.modules.decoders.decoder_helpers import Helper
from texar.torch.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.torch.utils import utils
from texar.torch.utils.beam_search import beam_search
from texar.torch.utils.types import MaybeList, MaybeTuple
from texar.torch.utils.utils import check_or_get_instance, get_function

__all__ = [
    'BasicRNNDecoderOutput',
    'AttentionRNNDecoderOutput',
    'BasicRNNDecoder',
    'AttentionRNNDecoder',
]


class BasicRNNDecoderOutput(NamedTuple):
    r"""The outputs of :class:`~texar.torch.modules.BasicRNNDecoder` that
    include both RNN outputs and sampled IDs at each step. This is also used to
    store results of all the steps after decoding the whole sequence.
    """
    logits: torch.Tensor
    r"""The outputs of RNN (at each step/of all steps) by applying the
    output layer on cell outputs. For example, in
    :class:`~texar.torch.modules.BasicRNNDecoder` with default hyperparameters,
    this is a :tensor:`Tensor` of shape ``[batch_size, max_time, vocab_size]``
    after decoding the whole sequence."""
    sample_id: torch.LongTensor
    r"""The sampled results (at each step/of all steps). For example, in
    :class:`~texar.torch.modules.BasicRNNDecoder` with decoding strategy of
    ``"train_greedy"``, this is a :tensor:`LongTensor` of shape
    ``[batch_size, max_time]`` containing the sampled token indices of all
    steps. Note that the shape of ``sample_id`` is different for different
    decoding strategy or helper. Please refer to
    :class:`~texar.torch.modules.Helper` for the detailed information."""
    cell_output: torch.Tensor
    r"""The output of RNN cell (at each step/of all steps). This contains the
    results prior to the output layer. For example, in
    :class:`~texar.torch.modules.BasicRNNDecoder` with default hyperparameters,
    this is a :tensor:`Tensor` of shape
    ``[batch_size, max_time, cell_output_size]`` after decoding the whole
    sequence."""


class AttentionRNNDecoderOutput(NamedTuple):
    r"""The outputs of :class:`~texar.torch.modules.AttentionRNNDecoder` that
    additionally includes attention results.
    """
    logits: torch.Tensor
    r"""The outputs of RNN (at each step/of all steps) by applying the
    output layer on cell outputs. For example, in
    :class:`~texar.torch.modules.AttentionRNNDecoder` with default
    hyperparameters, this is a :tensor:`Tensor` of shape
    ``[batch_size, max_time, vocab_size]`` after decoding the whole sequence."""
    sample_id: torch.LongTensor
    r"""The sampled results (at each step/of all steps). For example, in
    :class:`~texar.torch.modules.AttentionRNNDecoder` with decoding strategy of
    ``"train_greedy"``, this is a :tensor:`LongTensor` of shape
    ``[batch_size, max_time]`` containing the sampled token indices of all
    steps. Note that the shape of ``sample_id`` is different for different
    decoding strategy or helper. Please refer to
    :class:`~texar.torch.modules.Helper` for the detailed information."""
    cell_output: torch.Tensor
    r"""The output of RNN cell (at each step/of all steps). This contains the
    results prior to the output layer. For example, in
    :class:`~texar.torch.modules.AttentionRNNDecoder` with default
    hyperparameters, this is a :tensor:`Tensor` of shape
    ``[batch_size, max_time, cell_output_size]`` after decoding the whole
    sequence."""
    attention_scores: MaybeTuple[torch.Tensor]
    r"""A single or tuple of `Tensor(s)` containing the alignments emitted (at
    the previous time step/of all time steps) for each attention mechanism."""
    attention_context: torch.Tensor
    r"""The attention emitted (at the previous time step/of all time steps)."""


class BasicRNNDecoder(RNNDecoderBase[HiddenState, BasicRNNDecoderOutput]):
    r"""Basic RNN decoder.

    Args:
        input_size (int): Dimension of input embeddings.
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        token_embedder: An instance of :torch_nn:`Module`, or a function taking
            a :tensor:`LongTensor` ``tokens`` as argument. This is the embedder
            called in :meth:`embed_tokens` to convert input tokens to
            embeddings.
        token_pos_embedder: An instance of :torch_nn:`Module`, or a function
            taking two :tensor:`LongTensor`\ s ``tokens`` and ``positions`` as
            argument. This is the embedder called in :meth:`embed_tokens` to
            convert input tokens with positions to embeddings.

            .. note::
                Only one among :attr:`token_embedder` and
                :attr:`token_pos_embedder` should be specified. If neither is
                specified, you must subclass :class:`BasicRNNDecoder` and
                override :meth:`embed_tokens`.
        cell (RNNCellBase, optional): An instance of
            :class:`~texar.torch.core.cell_wrappers.RNNCellBase`. If `None`
            (default), a cell is created as specified in :attr:`hparams`.
        output_layer (optional): An instance of :torch_nn:`Module`. Apply to
            the RNN cell output to get logits. If `None`, a :torch_nn:`Linear`
            layer is used with output dimension set to :attr:`vocab_size`.
            Set ``output_layer`` to :func:`~texar.torch.core.identity` if you do
            not want to have an output layer after the RNN cell outputs.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    See :meth:`~texar.torch.modules.RNNDecoderBase.forward` for the inputs and
    outputs of the decoder. The decoder returns
    ``(outputs, final_state, sequence_lengths)``, where ``outputs`` is an
    instance of :class:`~texar.torch.modules.BasicRNNDecoderOutput`.

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

            # Create helper
            helper = decoder.create_helper(
                decoding_strategy='infer_sample',
                start_tokens=[data.vocab.bos_token_id]*100,
                end_token=data.vocab.eos.token_id,
                embedding=embedder)

            # Inference sample
            outputs, _, _ = decoder(
                helper=helerp,
                max_decoding_length=60)

            sample_text = tx.utils.map_ids_to_strs(
                outputs.sample_id, data.vocab)
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

        `"rnn_cell"`: dict
            A dictionary of RNN cell hyperparameters. Ignored if
            :attr:`cell` is given to the decoder constructor.
            The default value is defined in
            :func:`~texar.torch.core.default_rnn_cell_hparams`.

        `"max_decoding_length_train"`: int or None
            Maximum allowed number of decoding steps in training mode. If
            `None` (default), decoding is performed until fully done, e.g.,
            encountering the ``<EOS>`` token. Ignored if
            ``"max_decoding_length"`` is not `None` given when calling the
            decoder.

        `"max_decoding_length_infer"`: int or None
            Same as ``"max_decoding_length_train"`` but for inference mode.

        `"helper_train"`: dict
            The hyperparameters of the helper used in training.
            ``"type"`` can be a helper class, its name or module path, or a
            helper instance. If a class name is given, the class must be
            from module :mod:`texar.torch.modules`, or
            :mod:`texar.torch.custom`. This is used only when both
            ``"decoding_strategy"`` and ``"helper"`` arguments are `None` when
            calling the decoder. See
            :meth:`~texar.torch.modules.RNNDecoderBase.forward` for more
            details.

        `"helper_infer"`: dict
            Same as ``"helper_train"`` but during inference mode.

        `"name"`: str
            Name of the decoder.
            The default value is ``"basic_rnn_decoder"``.
        """
        hparams = RNNDecoderBase.default_hparams()
        hparams['name'] = 'basic_rnn_decoder'
        return hparams

    def step(self, helper: Helper, time: int, inputs: torch.Tensor,
             state: Optional[HiddenState]) \
            -> Tuple[BasicRNNDecoderOutput, HiddenState]:
        cell_outputs, cell_state = self._cell(inputs, state)
        logits = self._output_layer(cell_outputs)
        sample_ids = helper.sample(time=time, outputs=logits)
        next_state = cell_state
        outputs = BasicRNNDecoderOutput(logits, sample_ids, cell_outputs)
        return outputs, next_state

    def next_inputs(self, helper: Helper, time: int,
                    outputs: BasicRNNDecoderOutput) -> \
            Tuple[torch.Tensor, torch.ByteTensor]:
        finished, next_inputs = helper.next_inputs(
            self.embed_tokens, time, outputs.logits, outputs.sample_id)
        return next_inputs, finished

    @property
    def output_size(self):
        r"""Output size of one step.
        """
        return self._cell.hidden_size


class AttentionRNNDecoder(RNNDecoderBase[AttentionWrapperState,
                                         AttentionRNNDecoderOutput]):
    r"""RNN decoder with attention mechanism.

    Args:
        input_size (int): Dimension of input embeddings.
        encoder_output_size (int): The output size of the encoder cell.
        vocab_size (int): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        token_embedder: An instance of :torch_nn:`Module`, or a function taking
            a :tensor:`LongTensor` ``tokens`` as argument. This is the embedder
            called in :meth:`embed_tokens` to convert input tokens to
            embeddings.
        token_pos_embedder: An instance of :torch_nn:`Module`, or a function
            taking two :tensor:`LongTensor`\ s ``tokens`` and ``positions`` as
            argument. This is the embedder called in :meth:`embed_tokens` to
            convert input tokens with positions to embeddings.

            .. note::
                Only one among :attr:`token_embedder` and
                :attr:`token_pos_embedder` should be specified. If neither is
                specified, you must subclass :class:`AttentionRNNDecoder` and
                override :meth:`embed_tokens`.
        cell (RNNCellBase, optional): An instance of
            :class:`~texar.torch.core.cell_wrappers.RNNCellBase`. If `None`,
            a cell is created as specified in :attr:`hparams`.
        output_layer (optional): An output layer that transforms cell output
            to logits. This can be:

            - A callable layer, e.g., an instance of :torch_nn:`Module`.
            - A tensor. A dense layer will be created using the tensor
              as the kernel weights. The bias of the dense layer is determined
              by `hparams.output_layer_bias`. This can be used to tie the
              output layer with the input embedding matrix, as proposed in
              https://arxiv.org/pdf/1608.05859.pdf
            - `None`. A dense layer will be created based on :attr:`vocab_size`
              and `hparams.output_layer_bias`.
            - If no output layer after the cell output is needed, set
              `(vocab_size=None, output_layer=texar.torch.core.identity)`.
        cell_input_fn (callable, optional): A callable that produces RNN cell
            inputs. If `None` (default), the default is used:
            :python:`lambda inputs, attention:
            torch.cat([inputs, attention], -1)`,
            which concatenates regular RNN cell inputs with attentions.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    See :meth:`texar.torch.modules.RNNDecoderBase.forward` for the inputs and
    outputs of the decoder. The decoder returns
    `(outputs, final_state, sequence_lengths)`, where `outputs` is an instance
    of :class:`~texar.torch.modules.AttentionRNNDecoderOutput`.

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
                encoder_output_size=(self.encoder.cell_fw.hidden_size +
                                     self.encoder.cell_bw.hidden_size),
                input_size=dec_embedder.dim,
                vocab_size=data.target_vocab.size)
            outputs, _, _ = decoder(
                decoding_strategy='train_greedy',
                memory=enc_outputs,
                memory_sequence_length=data_batch['source_length'],
                inputs=dec_embedder(data_batch['target_text_ids']),
                sequence_length=data_batch['target_length']-1)
    """

    def __init__(self,
                 input_size: int,
                 encoder_output_size: int,
                 vocab_size: int,
                 token_embedder: Optional[TokenEmbedder] = None,
                 token_pos_embedder: Optional[TokenPosEmbedder] = None,
                 cell: Optional[RNNCellBase] = None,
                 output_layer: Optional[Union[nn.Module, torch.Tensor]] = None,
                 cell_input_fn: Optional[Callable[[torch.Tensor, torch.Tensor],
                                                  torch.Tensor]] = None,
                 hparams=None):

        super().__init__(
            input_size, vocab_size, token_embedder, token_pos_embedder,
            cell=cell, output_layer=output_layer, hparams=hparams)

        attn_hparams = self._hparams['attention']
        attn_kwargs = attn_hparams['kwargs'].todict()

        # Compute the correct input_size internally.
        if cell is None:
            if cell_input_fn is None:
                if attn_hparams["attention_layer_size"] is None:
                    input_size += encoder_output_size
                else:
                    input_size += attn_hparams["attention_layer_size"]
            else:
                if attn_hparams["attention_layer_size"] is None:
                    input_size = cell_input_fn(
                        torch.empty(input_size),
                        torch.empty(encoder_output_size)).shape[-1]
                else:
                    input_size = cell_input_fn(
                        torch.empty(input_size),
                        torch.empty(
                            attn_hparams["attention_layer_size"])).shape[-1]
            self._cell = layers.get_rnn_cell(input_size, self._hparams.rnn_cell)

        # Parse the `probability_fn` argument
        if 'probability_fn' in attn_kwargs:
            prob_fn = attn_kwargs['probability_fn']
            if prob_fn is not None and not callable(prob_fn):
                prob_fn = get_function(prob_fn, ['torch.nn.functional',
                                                 'texar.torch.core'])
            attn_kwargs['probability_fn'] = prob_fn

        # Parse `encoder_output_size` and `decoder_output_size` arguments
        if attn_hparams['type'] in ['BahdanauAttention',
                                    'BahdanauMonotonicAttention']:
            attn_kwargs.update({"decoder_output_size": self._cell.hidden_size})
        attn_kwargs.update({"encoder_output_size": encoder_output_size})

        attn_modules = ['texar.torch.core']
        # TODO: Support multiple attention mechanisms.
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
            if attn_hparams["attention_layer_size"] is None:
                self._output_layer = nn.Linear(
                    encoder_output_size,
                    vocab_size)
            else:
                self._output_layer = nn.Linear(
                    sum(attn_hparams["attention_layer_size"])
                    if isinstance(attn_hparams["attention_layer_size"], list)
                    else attn_hparams["attention_layer_size"],
                    vocab_size)

        attn_cell = AttentionWrapper(
            self._cell,
            self.attention_mechanism,
            cell_input_fn=self._cell_input_fn,
            **self._attn_cell_kwargs)

        self._cell: AttentionWrapper = attn_cell

        self.memory: Optional[torch.Tensor] = None
        self.memory_sequence_length: Optional[torch.LongTensor] = None

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.
        Common hyperparameters are the same as in
        :class:`~texar.torch.modules.BasicRNNDecoder`.
        :meth:`~texar.torch.modules.BasicRNNDecoder.default_hparams`.
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

        `"attention"`: dict
            Attention hyperparameters, including:

            `"type"`: str or class or instance
                The attention type. Can be an attention class, its name or
                module path, or a class instance. The class must be a subclass
                of ``AttentionMechanism``. See :ref:`attention-mechanism` for
                all supported attention mechanisms. If class name is given,
                the class must be from modules
                :mod:`texar.torch.core` or :mod:`texar.torch.custom`.

                Example:

                .. code-block:: python

                    # class name
                    "type": "LuongAttention"
                    "type": "BahdanauAttention"
                    # module path
                    "type": "texar.torch.core.BahdanauMonotonicAttention"
                    "type": "my_module.MyAttentionMechanismClass"
                    # class
                    "type": texar.torch.core.LuongMonotonicAttention
                    # instance
                    "type": LuongAttention(...)

            `"kwargs"`: dict
                keyword arguments for the attention class constructor.
                Arguments :attr:`memory` and
                :attr:`memory_sequence_length` should **not** be
                specified here because they are given to the decoder
                constructor. Ignored if "type" is an attention class
                instance. For example:

                .. code-block:: python

                    "type": "LuongAttention",
                    "kwargs": {
                        "num_units": 256,
                        "probability_fn": torch.nn.functional.softmax,
                    }

                Here `"probability_fn"` can also be set to the string name
                or module path to a probability function.

                `"attention_layer_size"`: int or None
                    The depth of the attention (output) layer. The context and
                    cell output are fed into the attention layer to generate
                    attention at each time step.
                    If `None` (default), use the context as attention at each
                    time step.

                `"alignment_history"`: bool
                    whether to store alignment history from all time steps
                    in the final output state. (Stored as a time major
                    `TensorArray` on which you must call `stack()`.)

                `"output_attention"`: bool
                    If `True` (default), the output at each time step is
                    the attention value. This is the behavior of Luong-style
                    attention mechanisms. If `False`, the output at each
                    time step is the output of `cell`.  This is the
                    behavior of Bahdanau-style attention mechanisms.
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

    def initialize(  # type: ignore
            self, helper: Helper,
            inputs: Optional[torch.Tensor],
            sequence_length: Optional[torch.LongTensor],
            initial_state: Optional[MaybeList[MaybeTuple[torch.Tensor]]]) -> \
            Tuple[torch.ByteTensor, torch.Tensor,
                  Optional[AttentionWrapperState]]:
        initial_finished, initial_inputs = helper.initialize(
            self.embed_tokens, inputs, sequence_length)
        if initial_state is None:
            state = None
        else:
            tensor = utils.get_first_in_structure(initial_state)
            assert tensor is not None
            tensor: torch.Tensor
            state = self._cell.zero_state(batch_size=tensor.size(0))
            state = state._replace(cell_state=initial_state)

        return initial_finished, initial_inputs, state

    def step(self, helper: Helper, time: int, inputs: torch.Tensor,
             state: Optional[AttentionWrapperState]) -> \
            Tuple[AttentionRNNDecoderOutput, AttentionWrapperState]:
        wrapper_outputs, wrapper_state = self._cell(
            inputs, state, self.memory, self.memory_sequence_length)
        # Essentially the same as in BasicRNNDecoder.step()

        logits = self._output_layer(wrapper_outputs)
        sample_ids = helper.sample(time=time, outputs=logits)

        attention_scores = wrapper_state.alignments
        attention_context = wrapper_state.attention
        outputs = AttentionRNNDecoderOutput(
            logits, sample_ids, wrapper_outputs,
            attention_scores, attention_context)
        next_state = wrapper_state

        return outputs, next_state

    def next_inputs(self, helper: Helper, time: int,
                    outputs: AttentionRNNDecoderOutput) -> \
            Tuple[torch.Tensor, torch.ByteTensor]:
        finished, next_inputs = helper.next_inputs(
            self.embed_tokens, time, outputs.logits, outputs.sample_id)
        return next_inputs, finished

    def forward(  # type: ignore
            self,
            memory: torch.Tensor,
            memory_sequence_length: Optional[torch.LongTensor] = None,
            inputs: Optional[torch.Tensor] = None,
            sequence_length: Optional[torch.LongTensor] = None,
            initial_state: Optional[MaybeList[MaybeTuple[torch.Tensor]]] = None,
            helper: Optional[Helper] = None,
            max_decoding_length: Optional[int] = None,
            impute_finished: bool = False,
            infer_mode: Optional[bool] = None,
            beam_width: Optional[int] = None,
            length_penalty: float = 0., **kwargs) \
            -> Union[Tuple[AttentionRNNDecoderOutput,
                           Optional[AttentionWrapperState], torch.LongTensor],
                     Dict[str, torch.Tensor]]:
        r"""Performs decoding.

        Implementation calls initialize() once and step() repeatedly on the
        Decoder object. Please refer to `tf.contrib.seq2seq.dynamic_decode`.

        See Also:
            Arguments of :meth:`create_helper`.

        Args:
            memory: The memory to query; usually the output of an RNN encoder.
                This tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length: (optional) Sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.
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
            helper (optional): An instance of
                :class:`~texar.torch.modules.Helper`
                that defines the decoding strategy. If given,
                ``decoding_strategy`` and helper configurations in
                :attr:`hparams` are ignored.
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
            infer_mode (optional): If not `None`, overrides mode given by
                `self.training`.
            beam_width (int): Set to use beam search. If given,
                :attr:`decoding_strategy` is ignored.
            length_penalty (float): Length penalty coefficient used in beam
                search decoding. Refer to https://arxiv.org/abs/1609.08144
                for more details.
                It should be larger if longer sentences are desired.
            **kwargs: Other keyword arguments for constructing helpers
                defined by ``hparams["helper_train"]`` or
                ``hparams["helper_infer"]``.

        Returns:

            - For **beam search** decoding, returns a ``dict`` containing keys
              ``"sample_id"`` and ``"log_prob"``.

                - ``"sample_id"`` is a :tensor:`LongTensor` of shape
                  ``[batch_size, max_time, beam_width]`` containing generated
                  token indexes. ``sample_id[:,:,0]`` is the highest-probable
                  sample.
                - ``"log_prob"`` is a :tensor:`Tensor` of shape
                  ``[batch_size, beam_width]`` containing the log probability
                  of each sequence sample.

            - For **"infer_greedy"** and **"infer_sample"** decoding or
              decoding with :attr:`helper`, returns
              a tuple `(outputs, final_state, sequence_lengths)`, where

                - **outputs**: an object containing the decoder output on all
                  time steps.
                - **final_state**: is the cell state of the final time step.
                - **sequence_lengths**: is an int Tensor of shape `[batch_size]`
                  containing the length of each sample.
        """
        # TODO: Add faster code path for teacher-forcing training.

        # Save memory and memory_sequence_length
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length

        # Maximum decoding length
        if max_decoding_length is None:
            if self.training:
                max_decoding_length = self._hparams.max_decoding_length_train
            else:
                max_decoding_length = self._hparams.max_decoding_length_infer
            if max_decoding_length is None:
                max_decoding_length = utils.MAX_SEQ_LENGTH

        # Beam search decode
        if beam_width is not None and beam_width > 1:
            if helper is not None:
                raise ValueError("Must not set 'beam_width' and 'helper' "
                                 "simultaneously.")

            start_tokens = kwargs.get('start_tokens')
            if start_tokens is None:
                raise ValueError("'start_tokens' must be specified when using"
                                 "beam search decoding.")

            batch_size = start_tokens.shape[0]
            state = self._cell.zero_state(batch_size)
            if initial_state is not None:
                state = state._replace(cell_state=initial_state)

            sample_id, log_prob = self.beam_decode(  # type: ignore
                start_tokens=start_tokens,
                end_token=kwargs.get('end_token'),
                initial_state=state,
                beam_width=beam_width,
                length_penalty=length_penalty,
                decode_length=max_decoding_length)

            # Release memory and memory_sequence_length in AttentionRNNDecoder
            self.memory = None
            self.memory_sequence_length = None

            # Release the cached memory in AttentionMechanism
            for attention_mechanism in self._cell.attention_mechanisms:
                attention_mechanism.clear_cache()

            return {"sample_id": sample_id,
                    "log_prob": log_prob}

        # Helper
        if helper is None:
            helper = self._create_or_get_helper(infer_mode, **kwargs)

        if (isinstance(helper, decoder_helpers.TrainingHelper) and
                (inputs is None or sequence_length is None)):
            raise ValueError("'input' and 'sequence_length' must not be None "
                             "when using 'TrainingHelper'.")

        # Initial state
        self._cell.init_batch()

        (outputs, final_state,
         sequence_lengths) = self.dynamic_decode(  # type: ignore
            helper, inputs, sequence_length, initial_state,
            max_decoding_length, impute_finished)

        # Release memory and memory_sequence_length in AttentionRNNDecoder
        self.memory = None
        self.memory_sequence_length = None

        # Release the cached memory in AttentionMechanism
        for attention_mechanism in self._cell.attention_mechanisms:
            attention_mechanism.clear_cache()

        return outputs, final_state, sequence_lengths

    def beam_decode(self,
                    start_tokens: torch.LongTensor,
                    end_token: int,
                    initial_state: AttentionWrapperState,
                    decode_length: int = 256,
                    beam_width: int = 5,
                    length_penalty: float = 0.6) \
            -> Tuple[torch.LongTensor, torch.Tensor]:

        def _prepare_beam_search(x):
            x = x.unsqueeze(1).repeat(1, beam_width, *([1] * (x.dim() - 1)))
            x = x.view(-1, *x.size()[2:])
            return x

        memory_beam_search = _prepare_beam_search(self.memory)
        memory_sequence_length_beam_search = _prepare_beam_search(
            self.memory_sequence_length)

        def _symbols_to_logits_fn(ids, state):
            batch_size = ids.size(0)
            step = ids.size(-1) - 1
            times = ids.new_full((batch_size,), step)
            inputs = self.embed_tokens(ids[:, -1], times)
            wrapper_outputs, wrapper_state = self._cell(
                inputs, state, memory_beam_search,
                memory_sequence_length_beam_search)
            logits = self._output_layer(wrapper_outputs)
            return logits, wrapper_state

        assert self._vocab_size is not None
        outputs, log_prob = beam_search(
            symbols_to_logits_fn=_symbols_to_logits_fn,
            initial_ids=start_tokens,
            beam_size=beam_width,
            decode_length=decode_length,
            vocab_size=self._vocab_size,
            alpha=length_penalty,
            states=initial_state,
            eos_id=end_token)

        # Ignores <BOS>
        outputs = outputs[:, :, 1:]
        # shape = [batch_size, seq_length, beam_width]
        outputs = outputs.permute((0, 2, 1))
        return outputs, log_prob

    @property
    def output_size(self):
        r"""Output size of one step.
        """
        return self._cell.output_size
