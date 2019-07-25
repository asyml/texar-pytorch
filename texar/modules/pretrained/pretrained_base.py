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
Base class for Pre-trained Modules.
"""

from typing import Optional

from texar.hyperparams import HParams
from texar.module_base import ModuleBase
from texar.modules.pretrained.bert_utils import (
    load_pretrained_bert, transform_bert_to_texar_config, init_bert_checkpoint)
from texar.modules.pretrained.gpt2_utils import (
    load_pretrained_gpt2, transform_gpt2_to_texar_config, init_gpt2_checkpoint)
from texar.modules.pretrained.xlnet_utils import (
    load_pretrained_xlnet, transform_xlnet_to_texar_config, init_xlnet_checkpoint)

__all__ = [
    "PretrainedMixin",
]


class PretrainedMixin(ModuleBase):
    r"""A mixin class for all pre-trained classes to inherit.
    """

    model_name: str
    pretrained_model_dir: Optional[str]

    def load_pretrained_config(self,
                               pretrained_model_name: Optional[str] = None,
                               cache_dir: Optional[str] = None,
                               hparams=None):
        r"""Load paths and configurations of the pre-trained model.

        Args:
            pretrained_model_name (optional): A str with the name
                of a pre-trained model to load. If `None`, will use the model
                name in :attr:`hparams`.
            cache_dir (optional): The path to a folder in which the
                pre-trained models will be cached. If `None` (default),
                a default directory will be used.
            hparams (dict or HParams, optional): Hyperparameters. Missing
                hyperparameter will be set to default values. See
                :meth:`default_hparams` for the hyperparameter structure
                and default values.
        """
        if not hasattr(self, "_hparams"):
            self._hparams = HParams(hparams, self.default_hparams())
        else:
            # Probably already parsed by subclasses. We rely on subclass
            # implementations to get this right.
            # As a sanity check, we require `hparams` to be `None` in this case.
            if hparams is not None:
                raise ValueError(
                    "`self._hparams` is already assigned, but `hparams` "
                    "argument is not None.")

        self.pretrained_model_dir = None

        if self.model_name == "BERT":
            load_func = load_pretrained_bert
            transform_func = transform_bert_to_texar_config
        elif self.model_name == "GPT2":
            load_func = load_pretrained_gpt2
            transform_func = transform_gpt2_to_texar_config
        elif self.model_name == "XLNet":
            load_func = load_pretrained_xlnet
            transform_func = transform_xlnet_to_texar_config
        else:
            raise ValueError("Could not find this pre-trained model.")

        if pretrained_model_name is None:
            pretrained_model_name = self._hparams.pretrained_model_name
        if pretrained_model_name is not None:
            self.pretrained_model_dir = load_func(
                pretrained_model_name, cache_dir)
            pretrained_model_hparams = transform_func(self.pretrained_model_dir)
            self._hparams = HParams(
                pretrained_model_hparams, self._hparams.todict())

    def init_pretrained_weights(self, *args, **kwargs):
        if self.model_name == "BERT":
            init_func = init_bert_checkpoint
        elif self.model_name == "GPT2":
            init_func = init_gpt2_checkpoint
        elif self.model_name == "XLNet":
            init_func = init_xlnet_checkpoint
        else:
            raise ValueError("Could not find this pre-trained model.")

        if self.pretrained_model_dir:
            init_func(self, self.pretrained_model_dir, *args, **kwargs)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        r"""Initialize parameters of the pre-trained model. This method is only
        called if pre-trained checkpoints are not loaded.
        """
        pass

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "pretrained_model_name": None,
                "name": "pretrained_base"
            }
        """
        return {
            'pretrained_model_name': None,
            'name': "pretrained_base",
            '@no_typecheck': ['pretrained_model_name']
        }

    def forward(self, inputs, *args, **kwargs):
        r"""Encodes the inputs and (optionally) conduct downstream prediction.

        Args:
            inputs: Inputs to the pre-trained module.
            *args: Other arguments.
            **kwargs: Keyword arguments.

        Returns:
            Encoding results or prediction results.
        """
        raise NotImplementedError
