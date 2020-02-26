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
ELMo encoder.
"""
import json
import os
import tempfile
import warnings

from typing import Any, Dict, List, Optional, Union

import torch

from torch.nn.modules import Dropout

from texar.torch.modules.encoders.encoder_base import EncoderBase
from texar.torch.modules.pretrained.elmo import PretrainedELMoMixin
from texar.torch.modules.pretrained.elmo_utils import (
    _ElmoBiLm, ScalarMix, remove_sentence_boundaries)

__all__ = [
    "ELMoEncoder",
]


class ELMoEncoder(EncoderBase, PretrainedELMoMixin):
    r"""ELMo model for encoding sequences. Please see
    :class:`~texar.torch.modules.PretrainedELMoMixin` for a brief description
    of ELMo.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``elmo-small``). Please refer to
            :class:`~texar.torch.modules.PretrainedELMoMixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """
    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        super().__init__(hparams=hparams)

        self.load_pretrained_config(pretrained_model_name, cache_dir)

        options_file = None
        weight_file = None
        tmp_dir = tempfile.TemporaryDirectory()
        if self.pretrained_model_dir is not None:
            info = list(os.walk(self.pretrained_model_dir))
            root, _, files = info[0]
            for file in files:
                if file.endswith('options.json'):
                    options_file = os.path.join(root, file)
                if file.endswith('weights.hdf5'):
                    weight_file = os.path.join(root, file)
        else:
            with open(os.path.join(tmp_dir.name, 'options.json'), "w") as fp:
                json.dump(self.hparams.encoder.todict(), fp)
            options_file = os.path.join(tmp_dir.name, 'options.json')

        assert options_file is not None
        self._elmo_lstm = _ElmoBiLm(
            options_file, weight_file,
            requires_grad=self.hparams.requires_grad,
            vocab_to_cache=self.hparams.vocab_to_cache)
        tmp_dir.cleanup()

        self._has_cached_vocab = self.hparams.vocab_to_cache is not None
        self._keep_sentence_boundaries = self.hparams.keep_sentence_boundaries
        self._dropout = Dropout(p=self.hparams.dropout)
        self._scalar_mixes: Any = []
        for k in range(self.hparams.num_output_representations):
            scalar_mix = ScalarMix(
                self._elmo_lstm.num_layers,
                do_layer_norm=self.hparams.do_layer_norm,
                initial_scalar_parameters=self.hparams.scalar_mix_parameters,
                trainable=self.hparams.scalar_mix_parameters is None)
            self.add_module("scalar_mix_{}".format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        * The encoder arch is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the encoder arch is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the encoder arch is defined by the
          configurations in `hparams` and weights are randomly initialized.

        .. code-block:: python

            {
                "pretrained_model_name": "elmo-small",
                "encoder": {
                    "lstm": {
                        "use_skip_connections": True,
                        "projection_dim": 128,
                        "cell_clip": 3,
                        "proj_clip": 3,
                        "dim": 1024,
                        "n_layers": 2
                    },
                    "char_cnn": {
                        "activation": "relu",
                        "filters": [[1, 32], [2, 32], [3, 64], [4, 128],
                                [5, 256], [6, 512], [7, 1024]],
                        "n_highway": 1,
                        "embedding": {
                            "dim": 16
                        },
                        "n_characters": 262,
                        "max_characters_per_token": 50
                    }
                },
                "num_output_representations": 2,
                "requires_grad": False,
                "do_layer_norm": False,
                "dropout": 0.5,
                "vocab_to_cache": None,
                "keep_sentence_boundaries": False,
                "scalar_mix_parameters": None,
                "name": "elmo_encoder",
            }

        Here:

        The default parameters are values for ELMo small model.

        `"pretrained_model_name"`: str or None
            The name of the pre-trained ELMo model. If None, the model
            will be randomly initialized.

        `"encoder"`: dict
            Hyperparameters for ELMo encoder.

        `"num_output_representations"`: int
            The number of ELMo representation to output with different linear
            weighted combination of the 3 layers (i.e., character-convnet
            output, the first LSTM output, the second LSTM output).

        `"requires_grad"`: bool
            If True, compute gradient of ELMo parameters for fine tuning.

        `"do_layer_norm"`: bool
            Should we apply layer normalization (passed to `ScalarMix`)?

        `"dropout"`: float
            The dropout to be applied to the ELMo representations.

        `"vocab_to_cache"`: List[string]
            A list of words to pre-compute and cache character convolutions
            for. If you use this option, ELMo expects that you pass word
            indices of shape `(batch_size, timesteps)` to forward, instead
            of character indices. If you use this option and pass a word which
            was not pre-cached, this will break.

        `"keep_sentence_boundaries"`: bool
            If True, the representation of the sentence boundary tokens are
            not removed.

        `"scalar_mix_parameters"`: List[float]
            If not `None`, use these scalar mix parameters to weight the
            representations produced by different layers. These mixing weights
            are not updated during training. The mixing weights here should be
            the unnormalized (i.e., pre-softmax) weights. So, if you wanted to
            use only the 1st layer of a 2-layer ELMo, you can set this to
            [-9e10, 1, -9e10 ].

        `"name"`: str
            Name of the module.
        """
        return {
            'pretrained_model_name': 'elmo-small',
            'encoder': {
                "lstm": {
                    "use_skip_connections": True,
                    "projection_dim": 128,
                    "cell_clip": 3,
                    "proj_clip": 3,
                    "dim": 1024,
                    "n_layers": 2
                },
                "char_cnn": {
                    "activation": "relu",
                    "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256],
                                [6, 512], [7, 1024]],
                    "n_highway": 1,
                    "embedding": {
                        "dim": 16
                    },
                    "n_characters": 262,
                    "max_characters_per_token": 50
                }
            },
            'num_output_representations': 2,
            'requires_grad': False,
            'do_layer_norm': False,
            'dropout': 0.5,
            'vocab_to_cache': None,
            'keep_sentence_boundaries': False,
            'scalar_mix_parameters': None,
            'name': 'elmo_encoder',
            '@no_typecheck': ['pretrained_model_name']
        }

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                word_inputs: Optional[torch.Tensor] = None) -> \
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        r"""Encodes the inputs.

        Args:
            inputs: Shape `[batch_size, max_time, 50]` of character ids
                representing the current batch.
            word_inputs: If you passed a cached vocab, you can in addition pass
                a tensor of shape `[batch_size, max_time]`, which represent
                word ids which have been pre-cached.

        Returns:
            A Dictionary with keys:

            - :attr:`elmo_representations`: A `num_output_representations` list
              of ELMo representations for the input sequence. Each
              representation is shape `[batch_size, max_time, embedding_dim]`

            - :attr:`mask`: Shape `(batch_size, timesteps)` long tensor
              with sequence mask.
        """
        # reshape the input if needed
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        if word_inputs is not None:
            original_word_size = word_inputs.size()
            if self._has_cached_vocab and len(original_word_size) > 2:
                reshaped_word_inputs = word_inputs.view(-1,
                                                        original_word_size[-1])
            elif not self._has_cached_vocab:
                warnings.warn(
                    "Word inputs were passed to ELMo but it does not have a "
                    "cached vocab.")
                reshaped_word_inputs = None  # type: ignore
            else:
                reshaped_word_inputs = word_inputs
        else:
            reshaped_word_inputs = word_inputs  # type: ignore

        # run the biLM
        bilm_output = self._elmo_lstm(reshaped_inputs, reshaped_word_inputs)
        layer_activations = bilm_output["activations"]
        mask_with_bos_eos = bilm_output["mask"]

        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, "scalar_mix_{}".format(i))
            representation_with_bos_eos = scalar_mix(layer_activations,
                                                     mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = \
                    remove_sentence_boundaries(
                        representation_with_bos_eos, mask_with_bos_eos)
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos
            representations.append(self._dropout(processed_representation))

        # reshape if necessary
        if word_inputs is not None and len(original_word_size) > 2:
            mask = processed_mask.view(original_word_size)
            elmo_representations = [
                representation.view(original_word_size + (-1,))
                for representation in representations]
        elif len(original_shape) > 3:
            mask = processed_mask.view(original_shape[:-1])
            elmo_representations = [
                representation.view(original_shape[:-1] + (-1,))
                for representation in representations]
        else:
            mask = processed_mask
            elmo_representations = representations

        return {"elmo_representations": elmo_representations, "mask": mask}

    @property
    def output_size(self):
        return self._elmo_lstm.get_output_dim()
