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
GPT2 decoder.
"""

from typing import Optional, Union, Tuple, Dict

import torch

from texar.core import layers
from texar.hyperparams import HParams
from texar.modules.pretrained.gpt2_utils import init_gpt2_checkpoint
from texar.modules.pretrained.pretrained_base import PretrainedBase
from texar.modules.embedders.embedders import WordEmbedder
from texar.modules.embedders.position_embedders import PositionEmbedder
from texar.modules.decoders.decoder_helpers import Helper
from texar.modules.decoders.transformer_decoders import (
    TransformerDecoder, TransformerDecoderOutput)

__all__ = [
    "GPT2Decoder",
]


class GPT2Decoder(PretrainedBase):
    r"""Raw GPT2 Transformer for decoding sequences.

    This module basically stacks
    :class:`~texar.modules.embedders.WordEmbedder`,
    :class:`~texar.modules.embedders.PositionEmbedder`,
    :class:`~texar.modules.encoders.TransformerDecoder`.

    This module supports the architecture first proposed
    in `(Radford et al.)` GPT2.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model to load selected in the list of:
            `117M`, `345M`. If `None`, will use the model name in
            :attr:`hparams`.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):

        self.model_name = "GPT2"

        super().__init__(pretrained_model_name=pretrained_model_name,
                         cache_dir=cache_dir,
                         hparams=hparams)

        if self.pretrained_model_dir:
            self._hparams = HParams(self.pretrained_model_hparams,
                                    self._hparams.todict())

        # Word embedding
        self.word_embedder = WordEmbedder(
            vocab_size=self._hparams.vocab_size,
            hparams=self._hparams.embed)

        # Position embedding
        self.position_embedder = PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed)

        # The GPT2 decoder (a TransformerDecoder)
        self.decoder = TransformerDecoder(
            vocab_size=self._hparams.vocab_size,
            output_layer=self.word_embedder.embedding,
            hparams=self._hparams.decoder)

        if self.pretrained_model_dir:
            init_gpt2_checkpoint(self, self.pretrained_model_dir)
        elif self._hparams.initializer:
            initialize = layers.get_initializer(self._hparams.initializer)
            assert initialize is not None
            # Do not re-initialize LayerNorm modules.
            for name, param in self.named_parameters():
                if name.split('.')[-1] == 'weight' and 'layer_norm' not in name:
                    initialize(param)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        * The decoder arch is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the encoder arch is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the encoder arch is defined by the
          configurations in `hparams` and weights are randomly initialized.

        .. code-block:: python

            {
                "pretrained_model_name": "117M",
                "vocab_size": 50257,
                "context_size": 1024,
                "embedding_size": 768,
                "embed": {
                    "dim": 768,
                    "name": "word_embeddings"
                },
                "position_size": 1024,
                "position_embed": {
                    "dim": 768,
                    "name": "position_embeddings"
                },

                "decoder": {
                    "dim": 768,
                    "num_blocks": 12,
                    "use_gpt_config": True,
                    "embedding_dropout": 0,
                    "residual_dropout": 0,
                    "multihead_attention": {
                        "use_bias": True,
                        "num_units": 768,
                        "num_heads": 12,
                        "dropout_rate": 0.0,
                        "output_dim": 768
                    },
                    "initializer": {
                        "type": "variance_scaling_initializer",
                        "kwargs": {
                            "factor": 1.0,
                            "mode": "FAN_AVG",
                            "uniform": True
                        }
                    },
                    "poswise_feedforward": {
                        "layers": [
                            {
                                "type": "Linear",
                                "kwargs": {
                                    "in_features": 768,
                                    "out_features": 3072,
                                    "bias": True
                                }
                            },
                            {
                                "type": "GPTGELU",
                                "kwargs": {}
                            },
                            {
                                "type": "Linear",
                                "kwargs": {
                                    "in_features": 3072,
                                    "out_features": 768,
                                    "bias": True
                                }
                            }
                        ],
                        "name": "ffn"
                    }
                },
                "initializer": None,
                "name": "gpt2_decoder",
            }

        Here:

        The default parameters are values for 117M GPT2 model.

        `"pretrained_model_name"`: str or None
            The name of the pre-trained GPT2 model. If None, the model
            will be randomly initialized.

        `"embed"`: dict
            Hyperparameters for word embedding layer.

        `"vocab_size"`: int
            The vocabulary size of `inputs` in `GPT2Model`.

        `"position_embed"`: dict
            Hyperparameters for position embedding layer.

        `"position_size"`:  int
            The maximum sequence length that this model might ever be used with.

        `"decoder"`: dict
            Hyperparameters for the TransformerDecoder.
            See :func:`~texar.modules.TransformerDecoder.default_harams`
            for details.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        `"name"`: str
            Name of the module.
        """
        return {
            'pretrained_model_name': '117M',
            'vocab_size': 50257,
            'context_size': 1024,
            'embedding_size': 768,
            'embed': {
                'dim': 768,
                'name': 'word_embeddings'
            },
            'position_size': 1024,
            'position_embed': {
                'dim': 768,
                'name': 'position_embeddings'
            },

            'decoder': {
                'dim': 768,
                'num_blocks': 12,
                'use_gpt_config': True,
                'embedding_dropout': 0,
                'residual_dropout': 0,
                'multihead_attention': {
                    'use_bias': True,
                    'num_units': 768,
                    'num_heads': 12,
                    "dropout_rate": 0.0,
                    'output_dim': 768
                },
                'initializer': {
                    'type': 'variance_scaling_initializer',
                    'kwargs': {
                        'factor': 1.0,
                        'mode': 'FAN_AVG',
                        'uniform': True
                    }
                },
                'poswise_feedforward': {
                    'layers': [
                        {
                            'type': 'Linear',
                            'kwargs': {
                                'in_features': 768,
                                'out_features': 3072,
                                'bias': True
                            }
                        },
                        {
                            'type': 'GPTGELU',
                            'kwargs': {}
                        },
                        {
                            'type': 'Linear',
                            'kwargs': {
                                'in_features': 3072,
                                'out_features': 768,
                                'bias': True
                            }
                        }
                    ],
                    'name': 'ffn'
                }
            },
            'initializer': None,
            'name': 'gpt2_decoder',
            '@no_typecheck': ['pretrained_model_name']
        }

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

        The interface is very similar to that of Transformer decoders
        (:class:`texar.modules.TransformerDecoder`). In particular,
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
           :class:`texar.modules.decoders.Helper`.
           This provides a superset of decoding strategies than above.
           The interface is the same as in RNN decoders.
           Please refer to :meth:`texar.modules.RNNDecoderBase.forward` for
           detailed usage and examples.

           Note that, here, though using a
           :class:`~texar.decoder.TrainingHelper` corresponding to the
           ``"train_greedy"`` strategy above, the implementation is *slower*
           than directly setting ``decoding_strategy="train_greedy"`` (though
           output results are the same).

           Argument :attr:`max_decoding_length` is optional.

        3. **Beam search**: set :attr:`beam_width` to use beam search decoding.
           Arguments :attr:`(start_tokens, end_token)` are required,
           and argument :attr:`max_decoding_length` is optional.

           .. warning::
               Beam search is not yet implemented. Setting :attr:`beam_width`
               to any value greater than 1 would raise a
               :exc:`NotImplementedError`

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
            inputs (optional): Input tensor for teacher forcing decoding, of
                shape ``[batch_size, target_max_time, emb_dim]`` containing the
                target sequence word embeddings. Used when
                :attr:`decoding_strategy` is set to ``"train_greedy"``.
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
                :class:`~texar.modules.Helper` will be ignored.
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
                :class:`texar.modules.decoders.Helper`
                that defines the decoding strategy. If given,
                ``decoding_strategy`` and helper configurations in
                :attr:`hparams` are ignored.
            infer_mode (optional): If not `None`, overrides mode given by
                :attr:`self.training`.

        Returns:

            - For **"train_greedy"** decoding, returns an instance of
              :class:`~texar.modules.TransformerDecoderOutput` which contains
              `sample_id` and `logits`.

            - For **"infer_greedy"** and **"infer_sample"** decoding or
              decoding with :attr:`helper`, returns
              a tuple ``(outputs, sequence_lengths)``, where ``outputs`` is an
              instance of :class:`~texar.modules.TransformerDecoderOutput` as
              in `"train_greedy"`, and ``sequence_lengths`` is a
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
        if inputs is not None:
            word_embeds = self.word_embedder(inputs)
            batch_size = inputs.shape[0]
            pos_length = inputs.new_full((batch_size,),
                                         inputs.shape[1],
                                         dtype=torch.int64)
            pos_embeds = self.position_embedder(sequence_length=pos_length)

            inputs = word_embeds + pos_embeds

        outputs = self.decoder(
            inputs=inputs,
            sequence_length=sequence_length,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            memory_attention_bias=memory_attention_bias,
            context=context,
            context_sequence_length=context_sequence_length,
            helper=helper,
            decoding_strategy=decoding_strategy,
            max_decoding_length=max_decoding_length,
            impute_finished=impute_finished,
            infer_mode=infer_mode,
            beam_width=beam_width,
            length_penalty=length_penalty,
            **kwargs)

        return outputs
