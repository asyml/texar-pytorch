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
Utils of ELMo Modules.
"""

import json
import os

from abc import ABC
from typing import Any, Dict

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedELMoMixin",
]

_ELMo_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"


class PretrainedELMoMixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the ELMo model.

    The ELMo model was proposed in
    `Deep contextualized word representations`_
    by `Peters et al.` from Allen Institute for Artificial Intelligence. It is
    a deep  bidirectional language model (biLM), which is pre-trained on a
    large text corpus.

    The available ELMo models are as follows:

      * ``elmo-small``: 13.6M parameters, trained on 800M tokens.
      * ``elmo-medium``: 28.0M parameters, trained on 800M tokens.
      * ``elmo-original``: 93.6M parameters, trained on 800M tokens.
      * ``elmo-original-5.5b``: 93.6M parameters, trained on 5.5B tokens.

    We provide the following ELMo classes:

      * :class:`~texar.torch.modules.ELMoEncoder` for text encoding.

    .. _`Deep contextualized word representations`:
        https://arxiv.org/abs/1802.05365
    """
    _MODEL_NAME = "ELMo"
    _MODEL2URL = {
        'elmo-small': [
            _ELMo_PATH + '2x1024_128_2048cnn_1xhighway/'
                         'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',
            _ELMo_PATH + '2x1024_128_2048cnn_1xhighway/'
                         'elmo_2x1024_128_2048cnn_1xhighway_options.json',
        ],
        'elmo-medium': [
            _ELMo_PATH + '2x2048_256_2048cnn_1xhighway/'
                         'elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5',
            _ELMo_PATH + '2x2048_256_2048cnn_1xhighway/'
                         'elmo_2x2048_256_2048cnn_1xhighway_options.json',
        ],
        'elmo-original': [
            _ELMo_PATH + '2x4096_512_2048cnn_2xhighway/'
                         'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
            _ELMo_PATH + '2x4096_512_2048cnn_2xhighway/'
                         'elmo_2x4096_512_2048cnn_2xhighway_options.json',
        ],
        'elmo-original-5.5b': [
            _ELMo_PATH + '2x4096_512_2048cnn_2xhighway_5.5B/'
                         'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',
            _ELMo_PATH + '2x4096_512_2048cnn_2xhighway_5.5B/'
                         'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
        ],
    }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None
        for file in files:
            if file.endswith('options.json'):
                config_path = os.path.join(root, file)
        if config_path is None:
            raise ValueError(f"Cannot find the config file in {cache_dir}")

        with open(config_path) as f:
            config_elmo = json.loads(f.read())

        return {'encoder': config_elmo}

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str, **kwargs):
        return
