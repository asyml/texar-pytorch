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
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch import nn

from texar.data.data_utils import maybe_download
from texar.hyperparams import HParams
from texar.module_base import ModuleBase
from texar.modules.pretrained.pretrained_utils import default_download_dir
from texar.utils.types import MaybeList

__all__ = [
    "PretrainedMixin",
]


class PretrainedMixin(ModuleBase, ABC):
    r"""A mixin class for all pre-trained classes to inherit.
    """

    _MODEL_NAME: str
    _MODEL2URL: Dict[str, MaybeList[str]]

    pretrained_model_dir: Optional[str]

    @classmethod
    def _download_checkpoint(cls, pretrained_model_name: str,
                             cache_dir: Optional[str] = None) -> str:
        r"""Download the specified pre-trained checkpoint, and return the
        directory in which the checkpoint is cached.

        Args:
            pretrained_model_name (str): Name of the model checkpoint.
            cache_dir (str, optional): Path to the cache directory. If `None`,
                uses the default directory given by
                :meth:`~texar.modules.default_download_dir`.

        Returns:
            Path to the cache directory.
        """
        if pretrained_model_name in cls._MODEL2URL:
            download_path = cls._MODEL2URL[pretrained_model_name]
        else:
            raise ValueError(
                f"Pre-trained model not found: {pretrained_model_name}")

        if cache_dir is None:
            cache_dir = default_download_dir(cls._MODEL_NAME)

        cache_path = os.path.join(cache_dir, pretrained_model_name)
        if not os.path.exists(cache_path):
            if isinstance(download_path, str):
                maybe_download(download_path, cache_path, extract=True)
                filename = download_path.split('/')[-1]
                folder_name = None
                for ext in ['.zip', '.tar.gz', '.tar.bz', '.tar']:
                    if filename.endswith(ext):
                        folder_name = filename[:-len(ext)]
                assert folder_name is not None
                folder_name = os.path.join(cache_path, folder_name)
                os.remove(os.path.join(cache_path, filename))
                for file in os.listdir(folder_name):
                    shutil.move(os.path.join(folder_name, file), cache_path)
                shutil.rmtree(folder_name)
            else:
                for path in download_path:
                    maybe_download(path, cache_path)
            print(f"Pre-trained {cls._MODEL_NAME} checkpoint "
                  f"{pretrained_model_name} cached to {cache_path}")
        else:
            print(f"Using cached pre-trained {cls._MODEL_NAME} checkpoint "
                  f"from {cache_path}.")

        return cache_path

    @classmethod
    @abstractmethod
    def _transform_config(cls, cache_dir: str) -> Dict[str, Any]:
        r"""Load the official config file and transform it into Texar-style
        hyperparameters.

        Args:
            cache_dir (str): Path to the cache directory.

        Returns:
            dict: Texar module hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_from_checkpoint(self, cache_dir: str, **kwargs):
        raise NotImplementedError

    def _name_to_variable(self, name: str) -> nn.Module:
        r"""Find the corresponding variable given the specified name.
        """
        pointer = self
        for m_name in name.split("."):
            if m_name.isdigit():
                num = int(m_name)
                pointer = pointer[num]  # type: ignore
            else:
                pointer = getattr(pointer, m_name)
        return pointer

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

        if pretrained_model_name is None:
            pretrained_model_name = self._hparams.pretrained_model_name
        if pretrained_model_name is not None:
            self.pretrained_model_dir = self._download_checkpoint(
                pretrained_model_name, cache_dir)
            pretrained_model_hparams = self._transform_config(
                self.pretrained_model_dir)
            self._hparams = HParams(
                pretrained_model_hparams, self._hparams.todict())

    def init_pretrained_weights(self, *args, **kwargs):
        if self.pretrained_model_dir:
            self._init_from_checkpoint(
                self.pretrained_model_dir, *args, **kwargs)
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
