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
XLNet Classifiers.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import itertools

import torch
import torch.nn as nn

from texar.modules.classifiers.classifier_base import ClassifierBase
from texar.modules.encoders.xlnet_encoder import XLNetEncoder
from texar.utils import utils


__all__ = [
    "XLNetClassifier",
]


class XLNetClassifier(ClassifierBase):
    r"""Classifier based on XLNet modules.

    Arguments are the same as in
    :class:`~texar.modules.XLNetEncoder`.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model to load selected in the list of:
            `xlnet-base-based`, `xlnet-large-cased`.
            If `None`, will use the model name in :attr:`hparams`.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):

        super().__init__(hparams)

        # Create the underlying encoder
        encoder_hparams = utils.dict_fetch(hparams,
                                           XLNetEncoder.default_hparams())
        if encoder_hparams is not None:
            encoder_hparams['name'] = None

        self._encoder = XLNetEncoder(
            pretrained_model_name=pretrained_model_name,
            cache_dir=cache_dir,
            hparams=encoder_hparams)

        if self._hparams.use_projection:
            self.projection = nn.Linear(self._encoder.output_size,
                                        self._encoder.output_size)
        self.dropout = nn.Dropout(self._hparams.dropout)

        if self._hparams.summary_type == 'last':
            self.summary_op = lambda output: output[-1]
        elif self._hparams.summary_type == 'first':
            self.summary_op = lambda output: output[0]
        elif self._hparams.summary_type == 'mean':
            self.summary_op = lambda output: torch.mean(output, dim=0)
        elif self._hparams.summary_type == 'attn':
            raise NotImplementedError
        else:
            raise ValueError(
                f"Unsupported summary type {self._hparams.summary_type}")

        # Create an additional classification layer if needed
        self.num_classes = self._hparams.num_classes
        if self.num_classes <= 0:
            self.hidden_to_logits = None
        else:
            self.hidden_to_logits = nn.Linear(self._encoder.output_size,
                                              self._hparams.num_classes)

        self.is_binary = (self.num_classes == 1) or \
                         (self.num_classes <= 0 and
                          self._hparams.hidden_dim == 1)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Same hyperparameters as in XLNetEncoder
                ...
                # (2) Additional hyperparameters
                "summary_type": "last",
                "use_projection": True,
                "num_classes": 2,
                "name": "xlnet_classifier",
            }

        Here:

        1. Same hyperparameters as in
            :class:`~texar.modules.XLNetEncoder`.
            See the :meth:`~texar.modules.XLNetEncoder.default_hparams`.
            An instance of XLNetEncoder is created for feature extraction.

        2. Additional hyperparameters:

            `summary_type`: str
                The summary type, one of:

                - If **last**, summary based on the output of the last time
                    step.
                - If **first**, summary based on the output of the first time
                    step.
                - If **mean**, summary based on the mean of the outputs of all
                    time steps.

            `use_projection`: bool
                If `True`, an additional `Linear` layer is added after the
                summary step.

            `num_classes`: int
                Number of classes:

                - If **> 0**, an additional `Linear`
                  layer is appended to the encoder to compute the logits over
                  classes.
                - If **<= 0**, no dense layer is appended. The number of
                  classes is assumed to be the final dense layer size of the
                  encoder.

            `name`: str
                Name of the classifier.
        """
        return {
            'pretrained_model_name': 'xlnet-base-cased',
            'untie_r': True,
            'num_layers': 12,
            'mem_len': 0,
            'reuse_len': 0,
            # layer
            'num_heads': 12,
            'hidden_dim': 768,
            'head_dim': 64,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'use_segments': True,
            # ffn
            'ffn_inner_dim': 3072,
            'activation': 'gelu',
            # embedding
            'vocab_size': 32000,
            'max_seq_len': 512,
            'initializer': None,
            "summary_type": "last",
            "use_projection": True,
            "num_classes": 2,
            "name": "xlnet_classifier",
            '@no_typecheck': ['pretrained_model_name'],
        }

    def param_groups(self,
                     lr: Optional[float] = None,
                     lr_layer_scale: float = 1.0,
                     decay_base_params: bool = False):
        r"""Create parameter groups for optimizers. When
        :attr:`lr_layer_decay_rate` is not 1.0, parameters from each layer form
        separate groups with different base learning rates.

        Args:
            lr (float): The learning rate. Can be omitted if
                :attr:`lr_layer_decay_rate` is 1.0.
            lr_layer_scale (float): Per-layer LR scaling rate. The `i`-th layer
                will be scaled by `lr_layer_scale ^ (num_layers - i - 1)`.
            decay_base_params (bool): If `True`, treat non-layer parameters
                (e.g. embeddings) as if they're in layer 0. If `False`, these
                parameters are not scaled.

        Returns:
            The parameter groups, used as the first argument for optimizers.
        """

        if lr_layer_scale != 1.0:
            if lr is None:
                raise ValueError(
                    "lr must be specified when lr_layer_decay_rate is not 1.0")

            def params_except_in(module: nn.Module,
                                 except_names: List[str]) \
                    -> Iterable[nn.Parameter]:
                return itertools.chain.from_iterable(
                    child.parameters() for name, child in
                    module.named_children()
                    if name not in except_names)

            num_layers = self._hparams.xlnet.num_layers
            base_group = {
                "params": params_except_in(
                    self.xlnet, ['attn_layers', 'ff_layers']),
                "lr": lr * (lr_layer_scale ** num_layers
                            if decay_base_params else 1.0)
            }
            fine_tune_group = {
                "params": params_except_in(self, ["xlnet"]),
                "lr": lr
            }
            param_groups = [base_group, fine_tune_group]
            for idx in range(num_layers):
                decay_rate = lr_layer_scale ** (num_layers - idx - 1)
                param_group = {
                    "params": [*self.xlnet.attn_layers[idx].parameters(),
                               *self.xlnet.ff_layers[idx].parameters()],
                    "lr": lr * decay_rate,
                }
                param_groups.append(param_group)
        else:
            param_groups = self.parameters()
        return param_groups

    def forward(self,  # type: ignore
                token_ids: torch.LongTensor,
                segment_ids: Optional[torch.LongTensor] = None,
                input_mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        r"""Feeds the inputs through the network and makes classification.

        Args:
            token_ids: Shape `(seq_len, batch_size)`.
            segment_ids: Shape `(seq_len, batch_size)`.
            input_mask: Float tensor of shape `(seq_len, batch_size)`. Note that
                positions with value 1 are masked out.

        Returns:
            A tuple `(logits, preds)`, containing the logits over classes and
            the predictions, respectively.

            - If ``num_classes`` == 1, ``logits`` and ``pred`` are both of
              shape ``[batch_size]``.
            - If ``num_classes`` > 1, ``logits`` is of shape
              ``[batch_size, num_classes]`` and ``pred`` is of shape
              ``[batch_size]``.
        """
        # output: (seq_len, batch_size, hidden_dim)
        output, _ = self._encoder(token_ids=token_ids,
                                  segment_ids=segment_ids,
                                  input_mask=input_mask)
        summary = self.summary_op(output)
        if self._hparams.use_projection:
            summary = torch.tanh(self.projection(summary))

        if self.hidden_to_logits is not None:
            # summary: (batch_size, hidden_dim)
            summary = self.dropout(summary)
            # logits: (batch_size, num_classes)
            logits = self.hidden_to_logits(summary)
        else:
            logits = summary

        # Compute predictions
        if self.is_binary:
            preds = (logits > 0).long()
            logits = torch.flatten(logits)
        else:
            preds = torch.argmax(logits, dim=-1)
        preds = torch.flatten(preds)

        return logits, preds
