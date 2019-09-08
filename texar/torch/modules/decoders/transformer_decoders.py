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
import warnings
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn

from texar.torch.core import layers
from texar.torch.modules.decoders.decoder_base import (
    DecoderBase, TokenEmbedder, TokenPosEmbedder, _make_output_layer)
from texar.torch.modules.decoders.decoder_helpers import (
    EmbeddingHelper, Helper)
from texar.torch.modules.encoders.multihead_attention import (
    Cache, MultiheadAttentionEncoder)
from texar.torch.modules.encoders.transformer_encoder import (
    default_transformer_poswise_net_hparams)
from texar.torch.modules.networks.networks import FeedForwardNetwork
from texar.torch.utils import transformer_attentions as attn
from texar.torch.utils.beam_search import beam_search
from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.utils import sequence_mask

__all__ = [
    'TransformerDecoderOutput',
    'TransformerDecoder',
]

EmbeddingFn = Callable[[torch.LongTensor, torch.LongTensor], torch.Tensor]


class TransformerDecoderOutput(NamedTuple):
    r"""The output of :class:`TransformerDecoder`.
    """
    logits: torch.Tensor
    r"""A :tensor:`Tensor` of shape ``[batch_size, max_time, vocab_size]``
    containing the logits."""
    sample_id: torch.LongTensor
    r"""A :tensor:`LongTensor` of shape ``[batch_size, max_time]``
    (or ``[batch_size, max_time, vocab_size]``) containing the sampled
    token indices. Note that the shape of ``sample_id`` is different for
    different decoding strategy or helper. Please refer to
    :class:`~texar.torch.modules.Helper` for the detailed information."""


class TransformerDecoder(DecoderBase[Cache, TransformerDecoderOutput]):
    r"""Transformer decoder that applies multi-head self-attention for
    sequence decoding.

    It is a stack of
    :class:`~texar.torch.modules.MultiheadAttentionEncoder`,
    :class:`~texar.torch.modules.FeedForwardNetwork`, and residual connections.

    Args:
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
                specified, you must subclass :class:`TransformerDecoder` and
                override :meth:`embed_tokens`.
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An output layer that transforms cell output
            to logits. This can be:

            - A callable layer, e.g., an instance of :torch_nn:`Module`.
            - A tensor. A :torch_nn:`Linear` layer will be created using the
              tensor as weights. The bias of the dense layer is determined
              by ``hparams.output_layer_bias``. This can be used to tie the
              output layer with the input embedding matrix, as proposed in
              https://arxiv.org/pdf/1608.05859.pdf.
            - `None`. A :torch_nn:`Linear` layer will be created based on
              :attr:`vocab_size` and ``hparams.output_layer_bias``.
            - If no output layer is needed at the end, set
              :attr:`vocab_size` to `None` and ``output_layer`` to
              :func:`~texar.torch.core.identity`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    .. document private functions
    """

    # State variables used during `dynamic_decode`. Assigned in `forward`.
    _state_max_decoding_length: int
    _state_context: Optional[torch.LongTensor]
    _state_context_sequence_length: Optional[torch.LongTensor]
    _state_cache: Cache

    def __init__(self,
                 token_embedder: Optional[TokenEmbedder] = None,
                 token_pos_embedder: Optional[TokenPosEmbedder] = None,
                 vocab_size: Optional[int] = None,
                 output_layer: Optional[Union[nn.Module, torch.Tensor]] = None,
                 hparams=None):
        super().__init__(
            token_embedder, token_pos_embedder,
            input_time_major=False, output_time_major=False, hparams=hparams)

        if token_pos_embedder is None and token_embedder is not None:
            warnings.warn(
                "Transformer models cannot capture positional information if "
                "no positional embedding is provided.")

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

        if self._hparams.use_gpt_config:
            eps = 1e-5
        else:
            eps = 1e-12

        for _ in range(self._hparams.num_blocks):
            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.self_attns.append(attn_module)
            self.self_attn_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=eps))

            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.enc_dec_attns.append(attn_module)
            self.end_dec_attn_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=eps))

            poswise_network = FeedForwardNetwork(
                hparams=self._hparams.poswise_feedforward)
            if (poswise_network.hparams.layers[-1]['kwargs']['out_features']
                    != self._hparams.dim):
                raise ValueError("The output dimension of "
                                 "FeedForwardNetwork should be equal "
                                 "to the dim of TransformerDecoder")
            self.poswise_networks.append(poswise_network)
            self.poswise_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=eps))

        self.final_layer_norm = nn.LayerNorm(self._input_size, eps=eps)
        self.embed_dropout = nn.Dropout(self._hparams.embedding_dropout)
        self.residual_dropout = nn.Dropout(self._hparams.residual_dropout)

        if self._hparams.initializer:
            # TODO: This might be different to what TensorFlow does
            initialize = layers.get_initializer(self._hparams.initializer)
            assert initialize is not None
            # Do not re-initialize LayerNorm modules.
            for name, param in self.named_parameters():
                if name.split(".")[-1] == "weight" and "layer_norm" not in name:
                    initialize(param)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # Same as in TransformerEncoder
                "num_blocks": 6,
                "dim": 512,
                "use_gpt_config": False,
                "embedding_dropout": 0.1,
                "residual_dropout": 0.1,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                "multihead_attention": {
                    'name': 'multihead_attention',
                    'num_units': 512,
                    'output_dim': 512,
                    'num_heads': 8,
                    'dropout_rate': 0.1,
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

        `"num_blocks"`: int
            Number of stacked blocks.

        `"dim"`: int
            Hidden dimension of the encoder.

        `"use_gpt_config"`: bool
            Whether to follow the `eps` setting of OpenAI GPT.

        `"embedding_dropout"`: float
            Dropout rate of the input word and position embeddings.

        `"residual_dropout"`: float
            Dropout rate of the residual connections.

        `"poswise_feedforward"`: dict
            Hyperparameters for a feed-forward network used in residual
            connections.
            Make sure the dimension of the output tensor is equal to ``dim``.

            See
            :func:`~texar.torch.modules.default_transformer_poswise_net_hparams`
            for details.

        `"multihead_attention"`: dict
            Hyperparameters for the multi-head attention strategy.
            Make sure the ``output_dim`` in this module is equal to ``dim``.

            See :class:`~texar.torch.modules.MultiheadAttentionEncoder`
            for details.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.

            See :func:`~texar.torch.core.get_initializer` for details.

        `"embedding_tie"`: bool
            Whether to use the word embedding matrix as the output layer
            that computes logits. If `False`, a new dense layer is created.

        `"output_layer_bias"`: bool
            Whether to use bias to the output layer.

        `"max_decoding_length"`: int
            The maximum allowed number of decoding steps.
            Set to a very large number of avoid the length constraint.
            Ignored if provided in :meth:`forward` or ``"train_greedy"``
            decoding is used.

        `"name"`: str
            Name of the module.
        """
        dim = 512
        return {
            'num_blocks': 6,
            'dim': dim,
            'use_gpt_config': False,
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

        :attr:`inputs` should be of shape ``[batch_size, dim]``.

        Returns:
            A tuple of logits and updated cache. Logits are of shape
            ``[batch_size, vocab_size]``.
        """
        outputs = self._self_attention_stack(
            inputs.unsqueeze(1), memory=cache['memory'], cache=cache)
        outputs = self._output_layer(outputs)
        outputs = outputs.squeeze(1)
        return outputs, cache

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
        (:class:`~texar.torch.modules.RNNDecoderBase`). In particular,
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
           :class:`~texar.torch.modules.Helper`.
           This provides a superset of decoding strategies than above.
           The interface is the same as in RNN decoders.
           Please refer to :meth:`texar.torch.modules.RNNDecoderBase.forward`
           for detailed usage and examples.

           Note that, here, though using a
           :class:`~texar.torch.modules.TrainingHelper` corresponding to the
           ``"train_greedy"`` strategy above, the implementation is *slower*
           than directly setting ``decoding_strategy="train_greedy"`` (though
           output results are the same).

           Argument :attr:`max_decoding_length` is optional.

        3. **Beam search**: set :attr:`beam_width` to use beam search decoding.
           Arguments :attr:`(start_tokens, end_token)` are required,
           and argument :attr:`max_decoding_length` is optional.

        Args:
            memory (optional): The memory to attend, e.g., the output of an RNN
                encoder. A :tensor:`Tensor` of shape
                ``[batch_size, memory_max_time, dim]``.
            memory_sequence_length (optional): A :tensor:`Tensor` of shape
                ``[batch_size]`` containing the sequence lengths for the batch
                entries in memory. Used to create attention bias of
                :attr:`memory_attention_bias` is not given. Ignored if
                :attr:`memory_attention_bias` is provided.
            memory_attention_bias (optional): A :tensor:`Tensor` of shape
                ``[batch_size, num_heads, memory_max_time, dim]``.
                An attention bias typically sets the value of a padding
                position to a large negative value for masking. If not given,
                :attr:`memory_sequence_length` is used to automatically
                create an attention bias.
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
            sequence_length (optional): A :tensor:`LongTensor` of shape
                ``[batch_size]``, containing the sequence length of
                :attr:`inputs`. Tokens beyond the respective sequence length are
                masked out.
                Used when :attr:`decoding_strategy` is set to
                ``"train_greedy"``.
            decoding_strategy (str): A string specifying the decoding
                strategy, including ``"train_greedy"``, ``"infer_greedy"``,
                ``"infer_sample"``.
                Different arguments are required based on the
                strategy. See above for details. Ignored if
                :attr:`beam_width` or :attr:`helper` is set.
            beam_width (int): Set to use beam search. If given,
                :attr:`decoding_strategy` is ignored.
            length_penalty (float): Length penalty coefficient used in beam
                search decoding. Refer to https://arxiv.org/abs/1609.08144
                for more details.
                It should be larger if longer sentences are desired.
            context (optional): An :tensor:`LongTensor` of shape
                ``[batch_size, length]``, containing the starting tokens for
                decoding. If context is set, ``start_tokens`` of the
                :class:`~texar.torch.modules.Helper` will be ignored.
            context_sequence_length (optional): Specify the length of context.
            max_decoding_length (int, optional): The maximum allowed number of
                decoding steps.
                If `None` (default), use ``"max_decoding_length"`` defined in
                :attr:`hparams`. Ignored in ``"train_greedy"`` decoding.
            impute_finished (bool): If `True`, then states for batch
                entries which are marked as finished get copied through and
                the corresponding outputs get zeroed out.  This causes some
                slowdown at each time step, but ensures that the final state
                and outputs have the correct values and that backprop ignores
                time steps that were marked as finished. Ignored in
                ``"train_greedy"`` decoding.
            helper (optional): An instance of
                :class:`~texar.torch.modules.Helper`
                that defines the decoding strategy. If given,
                ``decoding_strategy`` and helper configurations in
                :attr:`hparams` are ignored.
            infer_mode (optional): If not `None`, overrides mode given by
                :attr:`self.training`.

        Returns:

            - For **"train_greedy"** decoding, returns an instance of
              :class:`~texar.torch.modules.TransformerDecoderOutput` which
              contains `sample_id` and `logits`.

            - For **"infer_greedy"** and **"infer_sample"** decoding or
              decoding with :attr:`helper`, returns
              a tuple ``(outputs, sequence_lengths)``, where ``outputs`` is an
              instance of :class:`~texar.torch.modules.TransformerDecoderOutput`
              as in `"train_greedy"`, and ``sequence_lengths`` is a
              :tensor:`LongTensor` of shape ``[batch_size]`` containing the
              length of each sample.

            - For **beam search** decoding, returns a ``dict`` containing keys
              ``"sample_id"`` and ``"log_prob"``.

                - ``"sample_id"`` is a :tensor:`LongTensor` of shape
                  ``[batch_size, max_time, beam_width]`` containing generated
                  token indexes. ``sample_id[:,:,0]`` is the highest-probable
                  sample.
                - ``"log_prob"`` is a :tensor:`Tensor` of shape
                  ``[batch_size, beam_width]`` containing the log probability
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
            times = torch.arange(
                inputs.size(1), dtype=torch.long, device=inputs.device)
            times = times.unsqueeze(0).expand(inputs.size(0), -1)
            inputs = self.embed_tokens(inputs, times)
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

        if beam_width is None or beam_width == 1:  # Inference-like decoding
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
                pad_length = max_decoding_length - self._state_context.size(1)
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
            del cache  # not used

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
            end_token: int = kwargs.get('end_token')  # type: ignore

            # The output format is different when running beam search.
            sample_id, log_prob = self.beam_decode(
                start_tokens,
                end_token,
                embedding_fn=self.embed_tokens,
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
        r"""Forward through the stacked multi-head attentions.
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
        as different structure respectively. Specifically, for inference-like
        decoding, a simple list is used; for beam-search decoding, a
        :tensor:`Tensor` of shape ``[batch_size, current_steps, num_units]``
        is maintained, where ``current_steps`` is the number of steps currently
        decoded.
        """

        device = next(self.parameters()).device

        def _create_ta():
            return []

        def _create_empty_tensor():
            ret = torch.zeros(
                batch_size, 0, self._hparams.multihead_attention.num_units,
                dtype=torch.float, device=device)
            return ret

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

    def beam_decode(self, start_tokens: torch.LongTensor, end_token: int,
                    embedding_fn: Callable[
                        [torch.LongTensor, torch.LongTensor], torch.Tensor],
                    decode_length: int = 256, beam_width: int = 5,
                    length_penalty: float = 0.6) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        def _symbols_to_logits_fn(ids, cache):
            batch_size = ids.size(0)
            step = ids.size(-1) - 1
            times = ids.new_full((batch_size,), step)
            inputs = embedding_fn(ids[:, -1], times)
            return self._inputs_to_outputs(inputs, cache)

        assert self._vocab_size is not None

        outputs, log_prob = beam_search(
            _symbols_to_logits_fn,
            start_tokens,
            beam_width,
            decode_length,
            self._vocab_size,
            length_penalty,
            states=self._state_cache,
            eos_id=end_token)

        # Ignores <BOS>
        outputs = outputs[:, :, 1:]
        # shape = [batch_size, seq_length, beam_width]
        outputs = outputs.permute(0, 2, 1)
        return outputs, log_prob

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
            self.embed_tokens, inputs, sequence_length)
        state = initial_state or self._state_cache
        return initial_finished, initial_inputs, state

    def step(self, helper: Helper, time: int, inputs: torch.Tensor,
             state: Optional[Cache]) -> \
            Tuple[TransformerDecoderOutput, Cache]:
        assert state is not None
        outputs, state = self._inputs_to_outputs(inputs, state)
        sample_ids = helper.sample(time=time, outputs=outputs)
        if self._state_context is not None:
            assert self._state_context_sequence_length is not None
            sample_ids = torch.where(
                self._state_context_sequence_length > time,
                self._state_context[:, time],
                sample_ids)

        next_state = state
        outputs = TransformerDecoderOutput(
            logits=outputs,
            sample_id=sample_ids)
        return outputs, next_state

    def next_inputs(self, helper: Helper, time: int,
                    outputs: TransformerDecoderOutput) -> \
            Tuple[torch.Tensor, torch.ByteTensor]:
        finished, next_inputs = helper.next_inputs(
            self.embed_tokens, time, outputs.logits, outputs.sample_id)
        return next_inputs, finished

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
