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
Transformer decoder.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn

from texar import HParams, ModuleBase
from texar.core import layers
from texar.modules.networks import FeedForwardNetwork
from texar.modules.decoders import Helper
from texar.modules.decoders.decoder_base import DecoderBase, _make_output_layer
from texar.modules.decoders.decoder_helpers import EmbeddingHelper
from texar.modules.embedders import EmbedderBase
from texar.modules.encoders.multihead_attention import \
    Cache, MultiheadAttentionEncoder
from texar.modules.encoders.transformer_encoder import \
    default_transformer_poswise_net_hparams
# from texar.utils import beam_search
from texar.utils import get_instance, transformer_attentions as attn
from texar.utils.shapes import mask_sequences
from texar.utils.utils import sequence_mask

__all__ = [
    'TransformerDecoderOutput',
    'TransformerDecoder',
]


class TransformerDecoderOutput(NamedTuple):
    r"""The output of :class:`TransformerDecoder`.

    Attributes:
        logits: A float Tensor of shape
            `[batch_size, max_time, vocab_size]` containing the logits.
        sample_id: An int Tensor of shape `[batch_size, max_time]`
            containing the sampled token indexes.
    """
    logits: torch.Tensor
    sample_id: torch.LongTensor


class TransformerDecoder(DecoderBase[Cache, TransformerDecoderOutput]):
    r"""Transformer decoder that applies multi-head self-attention for
    sequence decoding.

    It is a stack of :class:`~texar.modules.encoders.MultiheadAttentionEncoder`,
    :class:`~texar.modules.FeedForwardNetwork`, and residual connections.

    Args:
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An output layer that transforms cell output
            to logits. This can be:

            - A callable layer, e.g., an instance of :class:`torch.nn.Module`.
            - A tensor. A :class:`~torch.nn.Linear` layer will be created using
            the tensor as weights. The bias of the dense layer is determined by
            `hparams.output_layer_bias`. This can be used to tie the output
            layer with the input embedding matrix, as proposed in
            https://arxiv.org/pdf/1608.05859.pdf
            - `None`. A dense layer will be created based on attr:`vocab_size`
            and `hparams.output_layer_bias`.
            - If no output layer is needed at the end, set
            `(vocab_size=None, output_layer=texar.core.identity)`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    .. document private functions
    .. automethod:: _build
    """

    # State variables used during `dynamic_decode`. Assigned in `forward`.
    _state_max_decoding_length: int
    _state_context: Optional[torch.LongTensor]
    _state_context_sequence_length: Optional[torch.LongTensor]
    _state_cache: Cache

    def __init__(self,
                 vocab_size: Optional[int] = None,
                 output_layer: Optional[Union[nn.Module, torch.Tensor]] = None,
                 hparams: Optional[HParams] = None):
        super().__init__(0, vocab_size,  # dummy value for input_size
                         input_time_major=False,
                         output_time_major=False, hparams=hparams)
        self._input_size = self._hparams.dim

        self._output_layer, self._vocab_size = _make_output_layer(
            output_layer, vocab_size, self._input_size,
            self._hparams.output_layer_bias)

        self.self_attns = nn.ModuleList()
        self.self_attn_layer_norm = nn.ModuleList()
        self.enc_dec_attns = nn.ModuleList()
        self.end_dec_attn_layer_norm = nn.ModuleList()
        self.poswise_networks = nn.ModuleList()
        self.poswise_layer_norm = nn.ModuleList()
        for i in range(self._hparams.num_blocks):
            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.self_attns.append(attn_module)
            self.self_attn_layer_norm.append(nn.LayerNorm(self._input_size))

            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.enc_dec_attns.append(attn_module)
            self.end_dec_attn_layer_norm.append(nn.LayerNorm(self._input_size))

            poswise_network = FeedForwardNetwork(
                hparams=self._hparams.poswise_feedforward)
            if (poswise_network._hparams.layers[-1]['kwargs']['out_features']
                    != self._hparams.dim):
                raise ValueError("The output dimension of "
                                 "FeedForwardNetwork should be equal "
                                 "to the dim of TransformerDecoder")
            self.poswise_networks.append(poswise_network)
            self.poswise_layer_norm.append(nn.LayerNorm(self._input_size))

        self.final_layer_norm = nn.LayerNorm(self._input_size)
        self.embed_dropout = nn.Dropout(self._hparams.embedding_dropout)
        self.residual_dropout = nn.Dropout(self._hparams.residual_dropout)

        if self._hparams.initializer:
            # TODO: This might be different to what TensorFlow does
            initialize = layers.get_initializer(self._hparams.initializer)
            assert initialize is not None
            for param in self.parameters():
                initialize(param)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # Same as in TransformerEncoder
                "num_blocks": 6,
                "dim": 512,
                "embedding_dropout": 0.1,
                "residual_dropout": 0.1,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                "multihead_attention": {
                    'name': 'multihead_attention',
                    'num_units': 512,
                    'output_dim': 512,
                    'num_heads': 8,
                    'dropout_rate': 0.1,
                    'output_dim': 512,
                    'use_bias': False,
                },
                "initializer": None,
                "name": "transformer_decoder"

                # Additional for TransformerDecoder
                "embedding_tie": True,
                "output_layer_bias": False,
                "max_decoding_length": int(1e10),
            }

        Here:
        "num_blocks" : int
            Number of stacked blocks.

        "dim" : int
            Hidden dimension of the encoder.

        "embedding_dropout": float
            Dropout rate of the input word and position embeddings.

        "residual_dropout" :  float
            Dropout rate of the residual connections.

        "poswise_feedforward" : dict
            Hyperparameters for a feed-forward network used in residual
            connections.
            Make sure the dimension of the output tensor is equal to `dim`.

            See :func:`~texar.modules.default_transformer_poswise_net_hparams`
            for details.

        "multihead_attention" : dict
            Hyperparameters for the multihead attention strategy.
            Make sure the `output_dim` in this module is equal to `dim`.

            See :func:`~texar.modules.MultiheadAttentionEncoder.default_hparams`
            for details.

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "embedding_tie" : bool
            Whether to use the word embedding matrix as the output layer
            that computes logits. If `False`, a new dense layer
            is created.

        "output_layer_bias" : bool
            Whether to use bias to the output layer.

        "max_decoding_length" : int
            The maximum allowed number of decoding steps.
            Set to a very large number of avoid the length constraint.
            Ignored if provided in :meth:`_build` or
            "train_greedy" decoding is used.

            Length penalty coefficient. Refer to
            https://arxiv.org/abs/1609.08144 for more details.

        "name" : str
            Name of the module.
        """
        dim = 512
        return {
            'num_blocks': 6,
            'dim': dim,
            'embedding_tie': True,
            'output_layer_bias': False,
            'max_decoding_length': int(1e10),
            'embedding_dropout': 0.1,
            'residual_dropout': 0.1,
            'poswise_feedforward': default_transformer_poswise_net_hparams(dim),
            'multihead_attention': {
                'name': 'multihead_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
            },
            'initializer': None,
            'name': "transformer_decoder",
        }

    def _inputs_to_outputs(self, inputs: torch.Tensor,
                           cache: Cache) -> Tuple[torch.Tensor, Cache]:
        r"""Returns the outputs of one decoding step (for example,
        the predicted logits of the next token).

        `inputs` should be of shape `[batch_size, dim]`.

        Returns outputs (i.e. logits) of shape `[batch_size, vocab_size]`
        and updated cache.
        """
        outputs = self._self_attention_stack(
            inputs.unsqueeze(1), memory=cache['memory'], cache=cache)
        outputs = self._output_layer(outputs)
        outputs = outputs.squeeze(1)
        return outputs, cache

    def _input_ids_to_outputs(self, input_ids: torch.LongTensor, step: int,
                              cache: Cache) -> Tuple[torch.Tensor, Cache]:
        """The function is called in beam-search decoding.

        `inputs` should be of shape `[batch_size]`.

        Returns outputs (i.e. logits) of shape `[batch_size, vocab_size]`
        and updated cache.
        """
        _batch_size = input_ids.size(0)
        times = input_ids.new_full((_batch_size,), step)
        inputs = self.embedding(input_ids, times)
        return self._inputs_to_outputs(inputs, cache)

    def forward(self,  # type: ignore
                inputs: Optional[torch.Tensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                memory: Optional[torch.Tensor] = None,
                memory_sequence_length: Optional[torch.LongTensor] = None,
                memory_attention_bias: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_sequence_length: Optional[torch.LongTensor] = None,
                helper: Optional[Helper] = None,
                decoding_strategy: str = 'train_greedy',
                max_decoding_length: Optional[int] = None,
                impute_finished: bool = False,
                infer_mode: Optional[bool] = None,
                beam_width: Optional[int] = None,
                length_penalty: float = 0.,
                **kwargs) \
            -> Union[
                TransformerDecoderOutput,
                Tuple[TransformerDecoderOutput, torch.LongTensor],
                Dict[str, torch.Tensor]]:
        r"""Performs decoding.

        The interface is very similar to that of RNN decoders
        (:meth:`texar.modules.RNNDecoderBase._build`). In particular,
        the function provides **3 ways** to specify the decoding method, with
        varying flexibility:

        1. The :attr:`decoding_strategy` argument.

            - **"train_greedy"**: decoding in teacher-forcing fashion (i.e.,
              feeding ground truth to decode the next step), and for each step
              sample is obtained by taking the `argmax` of logits.
              Argument :attr:`inputs` is required for this strategy.
              :attr:`sequence_length` is optional.
            - **"infer_greedy"**: decoding in inference fashion (i.e., feeding
              `generated` sample to decode the next step), and for each step
              sample is obtained by taking the `argmax` of logits.
              Arguments :attr:`(start_tokens, end_token)` are
              required for this strategy, and argument
              :attr:`max_decoding_length` is optional.
            - **"infer_sample"**: decoding in inference fashion, and for each
              step sample is obtained by `random sampling` from the logits.
              Arguments :attr:`(start_tokens, end_token)` are required for this
              strategy, and argument :attr:`max_decoding_length` is optional.

          This argument is used only when arguments :attr:`helper` and
          :attr:`beam_width` are both `None`.

        2. The :attr:`helper` argument: An instance of subclass of
           :tf_main:`tf.contrib.seq2seq.Helper <contrib/seq2seq/Helper>`.
           This provides a superset of decoding strategies than above.
           The interface is the same as in RNN decoders.
           Please refer to :meth:`texar.modules.RNNDecoderBase._build` for
           detailed usage and examples.

           Note that, here, though using a :tf_main:`TrainingHelper
           <contrib/seq2seq/TrainingHelper>` corresponding to the
           "train_greedy" strategy above, the implementation is *slower* than
           directly setting `decoding_strategy="train_greedy"` (though the
           output results are the same).

           Argument :attr:`max_decoding_length` is optional.

        3. **Beam search**: set :attr:`beam_width` to use beam search decoding.
           Arguments :attr:`(start_tokens, end_token)` are required,
           and argument :attr:`max_decoding_length` is optional.

        Args:
            memory (optional): The memory to attend, e.g., the output of an RNN
                encoder. A Tensor of shape `[batch_size, memory_max_time, dim]`.
            memory_sequence_length (optional): A Tensor of shape `[batch_size]`
                containing the sequence lengths for the batch entries in
                memory. Used to create attention bias of
                :attr:`memory_attention_bias` is not given. Ignored if
                `memory_attention_bias` is provided.
            memory_attention_bias (optional): A Tensor of shape
                `[batch_size, num_heads, memory_max_time, dim]`.
                An attention bias typically sets the value of a padding
                position to a large negative value for masking. If not given,
                :attr:`memory_sequence_length` is used to automatically
                create an attention bias.
            inputs (optional): Input tensor for teacher forcing decoding, of
                shape `[batch_size, target_max_time, emb_dim]` containing the
                target sequence word embeddings.
                Used when :attr:`decoding_strategy` is set to "train_greedy".
            sequence_length (optional): A Tensor of shape `[batch_size]`,
                containing the sequence length of :attr:`inputs`.
                Tokens beyond the respective sequence length are masked out.
                Used when :attr:`decoding_strategy` is set to
                "train_greedy".
            decoding_strategy (str): A string specifying the decoding
                strategy, including "train_greedy", "infer_greedy",
                "infer_sample".
                Different arguments are required based on the
                strategy. See above for details. Ignored if
                :attr:`beam_width` or :attr:`helper` is set.
            beam_width (int): Set to use beam search. If given,
                :attr:`decoding_strategy` is ignored.
            length_penalty (float): Length penalty coefficient used in beam
                search decoding. Refer to https://arxiv.org/abs/1609.08144
                for more details.
                It Should be larger if longer sentences are wanted.
            start_tokens (optional): An int Tensor of shape `[batch_size]`,
                containing the start tokens.
                Used when :attr:`decoding_strategy` = "infer_greedy" or
                "infer_sample", or :attr:`beam_width` is set.
                Ignored when context is set.
            end_token (optional): An int 0D Tensor, the token that marks end
                of decoding.
                Used when :attr:`decoding_strategy` = "infer_greedy" or
                "infer_sample", or :attr:`beam_width` is set.
            context (optional): An int Tensor of shape `[batch_size, length]`,
                containing the starting tokens for decoding.
                If context is set, the start_tokens will be ignored.
            context_sequence_length (optional): specify the length of context.
            softmax_temperature (optional): A float 0D Tensor, value to divide
                the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples. Must > 0. If `None`,
                1.0 is used.
                Used when :attr:`decoding_strategy` = "infer_sample"`.
            max_decoding_length (optional): An int scalar Tensor indicating
                the maximum allowed number of decoding steps.
                If `None` (default), use "max_decoding_length" defined in
                :attr:`hparams`. Ignored in "train_greedy" decoding.
            impute_finished (bool): If `True`, then states for batch
                entries which are marked as finished get copied through and
                the corresponding outputs get zeroed out.  This causes some
                slowdown at each time step, but ensures that the final state
                and outputs have the correct values and that backprop ignores
                time steps that were marked as finished. Ignored in
                "train_greedy" decoding.
            helper (optional): An instance of
                :tf_main:`Helper <contrib/seq2seq/Helper>` that defines the
                decoding strategy. If given, :attr:`decoding_strategy` is
                ignored.
            infer_mode (optional): If not `None`, overrides mode given by
                `self.training`.

        Returns:

            - For **"train_greedy"** decoding, returns an instance of \
            :class:`~texar.modules.TransformerDecoderOutput` which contains\
            `sample_id` and `logits`.

            - For **"infer_greedy"** and **"infer_sample"** decoding or\
            decoding with :attr:`helper`, returns\
            a tuple `(outputs, sequence_lengths)`, where `outputs` is an \
            instance of :class:`~texar.modules.TransformerDecoderOutput` as\
            in "train_greedy", and `sequence_lengths` is a Tensor of shape\
            `[batch_size]` containing the length of each sample.

            - For **beam search** decoding, returns a `dict` containing keys\
            "sample_id" and "log_prob".

                - **"sample_id"** is an int Tensor of shape \
                `[batch_size, max_time, beam_width]` containing generated\
                token indexes. `sample_id[:,:,0]` is the highest-probable \
                sample.
                - **"log_prob"** is a float Tensor of shape \
                `[batch_size, beam_width]` containing the log probability \
                of each sequence sample.
        """

        if memory is not None:
            if memory_attention_bias is None:
                if memory_sequence_length is None:
                    raise ValueError(
                        "`memory_sequence_length` is required if "
                        "`memory_attention_bias` is not given.")

                enc_padding = 1 - sequence_mask(
                    memory_sequence_length, memory.size(1),
                    dtype=torch.float32)
                memory_attention_bias = attn.attention_bias_ignore_padding(
                    enc_padding)

        # record the context, which will be used in step function
        # for dynamic_decode
        if context is not None:
            if context_sequence_length is None:
                raise ValueError("'context_sequence_length' must not be None"
                                 "when 'context' is specified.")
            self._state_context = context[:, 1:]
            self._state_context_sequence_length = context_sequence_length - 1
        else:
            self._state_context = None
            self._state_context_sequence_length = None

        # Faster code path for teacher-forcing training
        if (helper is None and beam_width is None and
                decoding_strategy == 'train_greedy'):
            if inputs is None:
                raise ValueError("'input' must not be none "
                                 "when using 'train_greedy' decoding strategy.")
            if sequence_length is not None:
                inputs = mask_sequences(inputs, sequence_length)

            decoder_self_attention_bias = (
                attn.attention_bias_lower_triangle(inputs.size(1)))

            decoder_output = self._self_attention_stack(
                inputs, memory, decoder_self_attention_bias,
                memory_attention_bias, cache=None)
            logits = self._output_layer(decoder_output)
            sample_id = torch.argmax(logits, dim=-1)

            return TransformerDecoderOutput(logits, sample_id)

        # Inference code path.
        if max_decoding_length is None:
            max_decoding_length = self._hparams.max_decoding_length

        self._state_max_decoding_length = max_decoding_length

        if beam_width is None:  # Inference-like decoding
            # Prepare helper
            if helper is None:
                kwargs.update(decoding_strategy=decoding_strategy)
                if context is not None:
                    kwargs.update(start_tokens=context[:, 0])
                helper = self._create_or_get_helper(infer_mode, **kwargs)
            assert isinstance(helper, EmbeddingHelper)

            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=False, batch_size=helper.batch_size)
            if context is not None:
                assert self._state_context is not None
                pad_length = \
                    max_decoding_length - self._state_context.size(1)
                if pad_length > 0:
                    self._state_context = torch.cat((
                        self._state_context,
                        self._state_context.new_zeros(
                            self._state_context.size(0), pad_length)
                    ), dim=1)

            outputs, cache, sequence_lengths = self.dynamic_decode(
                helper, inputs=None, sequence_length=None,
                initial_state=None, max_decoding_length=max_decoding_length,
                impute_finished=impute_finished)

            if context is not None:
                # Here the length of sample_id will be larger than that
                # of logit by 1, because there will be a additional
                # start_token in the returned sample_id.
                # the start_id should be the first token of the
                # given context
                start_tokens = context[:, 0]
                outputs = TransformerDecoderOutput(
                    logits=outputs.logits,
                    sample_id=torch.cat([
                        start_tokens.unsqueeze(1),
                        outputs.sample_id
                    ], dim=1))
                sequence_lengths = sequence_lengths + 1

            return outputs, sequence_lengths

        else:  # Beam-search decoding
            # Ignore `decoding_strategy` and # assume `helper` is not set.
            if helper is not None:
                raise ValueError("Must not set 'beam_width' and 'helper' "
                                 "simultaneously.")
            if context is not None:
                start_tokens = context[:, 0]
            else:
                if 'start_tokens' not in kwargs:
                    raise ValueError(
                        "'start_tokens' must be specified when using"
                        "beam search decoding.")
                start_tokens = kwargs['start_tokens']
            _batch_size = start_tokens.size(0)
            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=True,
                batch_size=_batch_size)

            # The output format is different when running beam search
            sample_id, log_prob = self._beam_decode(
                start_tokens,
                kwargs.get('end_token'),
                beam_width=beam_width,
                length_penalty=length_penalty,
                decode_length=max_decoding_length)

            return {
                'sample_id': sample_id,
                'log_prob': log_prob
            }

    def _self_attention_stack(
            self, inputs: torch.Tensor,
            memory: Optional[torch.Tensor],
            decoder_self_attention_bias: Optional[torch.Tensor] = None,
            memory_attention_bias: Optional[torch.Tensor] = None,
            cache: Optional[Cache] = None) -> torch.Tensor:
        r"""Stacked multihead attention module.
        """
        inputs = self.embed_dropout(inputs)
        if cache is not None:
            if memory is not None:
                memory_attention_bias = cache['memory_attention_bias']
        else:
            assert decoder_self_attention_bias is not None

        x = inputs
        for i in range(self._hparams.num_blocks):
            layer_cache = cache['layers'][i] if cache is not None else None

            selfatt_output = self.self_attns[i](
                queries=self.self_attn_layer_norm[i](x),
                memory=None,
                memory_attention_bias=decoder_self_attention_bias,
                cache=layer_cache)
            x = x + self.residual_dropout(selfatt_output)

            if memory is not None:
                encdec_output = self.enc_dec_attns[i](
                    queries=self.end_dec_attn_layer_norm[i](x),
                    memory=memory,
                    memory_attention_bias=memory_attention_bias)
                x = x + self.residual_dropout(encdec_output)

            sub_output = self.poswise_networks[i](self.poswise_layer_norm[i](x))
            x = x + self.residual_dropout(sub_output)

        return self.final_layer_norm(x)

    def _init_cache(self, memory: Optional[torch.Tensor],
                    memory_attention_bias: Optional[torch.Tensor],
                    beam_search_decoding: bool,
                    batch_size: int) -> Cache:
        r"""Returns an initialized cache.

        In order to support both inference-like decoding and beam-search
        decoding, the elements of each layer must be initialized and extended
        as different structure respectively. Specifically, when inference-like
        decoding, tf.TensorArray is used, which satisfies the shape consistency
        check in the while-loop in tf.contrib.seq2seq.dynamic_decode. When
        beam-search decoding, a tf.Tensor of shape
        `[batch_size, current_steps, num_units]` is maintained, where
        `current_steps` is the number of steps currently decoded.
        """

        def _create_ta():
            return []

        def _create_empty_tensor():
            return torch.zeros(
                batch_size, 0, self._hparams.multihead_attention.num_units,
                dtype=torch.float)

        _create_fn = (_create_empty_tensor if beam_search_decoding
                      else _create_ta)

        cache: Cache = {
            'memory': memory,
            'memory_attention_bias': memory_attention_bias,
            'layers': [{
                'keys': _create_fn(),
                'values': _create_fn(),
            } for _ in range(self._hparams.num_blocks)],
        }

        return cache

    def _beam_decode(self,
                     start_tokens,
                     end_token,
                     decode_length=256,
                     beam_width=5,
                     length_penalty=0.6):
        # def _symbols_to_logits_fn(ids, step, cache):
        #     return self._inputs_to_outputs(
        #         self._embedding(ids[:, -1]), step, cache)
        #
        # outputs, log_prob = beam_search.beam_search(
        #     _symbols_to_logits_fn,
        #     start_tokens,
        #     beam_width,
        #     decode_length,
        #     self._vocab_size,
        #     length_penalty,
        #     states=self._state_cache,
        #     eos_id=end_token)
        #
        # # Ignores <BOS>
        # outputs = outputs[:, :, 1:]
        # # shape = [batch_size, seq_length, beam_width]
        # outputs = outputs.permute([0, 2, 1])
        # return (outputs, log_prob)
        raise NotImplementedError

    @property
    def output_size(self) -> int:
        r"""Output size of one step.
        """
        return self._input_size

    def initialize(self, helper: Helper, inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor],
                   initial_state: Optional[Cache]) \
            -> Tuple[torch.ByteTensor, torch.Tensor, Cache]:
        initial_finished, initial_inputs = helper.initialize(
            inputs, sequence_length)
        state = initial_state or self._state_cache
        return (initial_finished, initial_inputs, state)

    def step(self, helper: Helper, time: int,
             inputs: torch.Tensor, state: Optional[Cache]) \
            -> Tuple[TransformerDecoderOutput, Cache,
                     torch.Tensor, torch.ByteTensor]:
        assert state is not None
        outputs, state = self._inputs_to_outputs(inputs, state)
        sample_ids = helper.sample(time=time, outputs=outputs)
        if self._state_context is not None:
            assert self._state_context_sequence_length is not None
            sample_ids = torch.where(
                self._state_context_sequence_length > time,
                self._state_context[:, time],
                sample_ids)

        if time + 1 == self._state_max_decoding_length:
            # Maximum decoding length reached, mark all batches as finished.
            # This requires special handling because performing lookup on
            # position embeddings with `time + 1` may result in IndexError.
            finished = torch.ones_like(sample_ids, dtype=torch.uint8)
            # Since `next_inputs` will not be used, simply create a null tensor.
            next_inputs = torch.empty(0)
        else:
            finished, next_inputs = helper.next_inputs(
                time=time, outputs=outputs, sample_ids=sample_ids)
        next_state = state
        outputs = TransformerDecoderOutput(
            logits=outputs,
            sample_id=sample_ids)
        return outputs, next_state, next_inputs, finished

    def finalize(self,  # type: ignore
                 outputs: TransformerDecoderOutput,
                 final_state: Optional[Cache],
                 sequence_lengths: torch.LongTensor) \
            -> Tuple[TransformerDecoderOutput, Optional[Cache]]:
        # Clear state variables at end of decoding.
        del self._state_max_decoding_length
        del self._state_context
        del self._state_context_sequence_length
        del self._state_cache

        return super().finalize(outputs, final_state, sequence_lengths)
