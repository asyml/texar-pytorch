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
BERT classifier.
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from texar.torch.core.layers import get_initializer
from texar.torch.hyperparams import HParams
from texar.torch.modules.classifiers.classifier_base import ClassifierBase
from texar.torch.modules.encoders.bert_encoder import BERTEncoder
from texar.torch.modules.pretrained.bert import PretrainedBERTMixin
from texar.torch.utils.utils import dict_fetch

__all__ = [
    "BERTClassifier"
]


class BERTClassifier(ClassifierBase, PretrainedBERTMixin):
    r"""Classifier based on BERT modules. Please see
    :class:`~texar.torch.modules.PretrainedBERTMixin` for a brief description
    of BERT.

    This is a combination of the
    :class:`~texar.torch.modules.BERTEncoder` with a classification
    layer. Both step-wise classification and sequence-level classification
    are supported, specified in :attr:`hparams`.

    Arguments are the same as in
    :class:`~texar.torch.modules.BERTEncoder`.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``bert-base-uncased``). Please refer to
            :class:`~texar.torch.modules.PretrainedBERTMixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.

    .. document private functions
    """
    _ENCODER_CLASS = BERTEncoder

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):

        super().__init__(hparams=hparams)

        # Create the underlying encoder
        encoder_hparams = dict_fetch(hparams,
                                     self._ENCODER_CLASS.default_hparams())

        self._encoder = self._ENCODER_CLASS(
            pretrained_model_name=pretrained_model_name,
            cache_dir=cache_dir,
            hparams=encoder_hparams)

        # Create a dropout layer
        self._dropout_layer = nn.Dropout(self._hparams.dropout)

        # Create an additional classification layer if needed
        self.num_classes = self._hparams.num_classes
        if self.num_classes <= 0:
            self._logits_layer = None
        else:
            logit_kwargs = self._hparams.logit_layer_kwargs
            if logit_kwargs is None:
                logit_kwargs = {}
            elif not isinstance(logit_kwargs, HParams):
                raise ValueError("hparams['logit_layer_kwargs'] "
                                 "must be a dict.")
            else:
                logit_kwargs = logit_kwargs.todict()

            if self._hparams.clas_strategy == 'all_time':
                self._logits_layer = nn.Linear(
                    self._encoder.output_size *
                    self._hparams.max_seq_length,
                    self.num_classes,
                    **logit_kwargs)
            else:
                self._logits_layer = nn.Linear(
                    self._encoder.output_size, self.num_classes,
                    **logit_kwargs)

        if self._hparams.initializer:
            initialize = get_initializer(self._hparams.initializer)
            assert initialize is not None
            if self._logits_layer:
                initialize(self._logits_layer.weight)
                if self._logits_layer.bias:
                    initialize(self._logits_layer.bias)

        self.is_binary = (self.num_classes == 1) or \
                         (self.num_classes <= 0 and
                          self._hparams.encoder.dim == 1)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Same hyperparameters as in BertEncoder
                ...
                # (2) Additional hyperparameters
                "num_classes": 2,
                "logit_layer_kwargs": None,
                "clas_strategy": "cls_time",
                "max_seq_length": None,
                "dropout": 0.1,
                "name": "bert_classifier"
            }

        Here:

        1. Same hyperparameters as in
           :class:`~texar.torch.modules.BERTEncoder`.
           See the :meth:`~texar.torch.modules.BERTEncoder.default_hparams`.
           An instance of BERTEncoder is created for feature extraction.

        2. Additional hyperparameters:

            `"num_classes"`: int
                Number of classes:

                - If **> 0**, an additional `Linear`
                  layer is appended to the encoder to compute the logits over
                  classes.
                - If **<= 0**, no dense layer is appended. The number of
                  classes is assumed to be the final dense layer size of the
                  encoder.

            `"logit_layer_kwargs"`: dict
                Keyword arguments for the logit Dense layer constructor,
                except for argument "units" which is set to `num_classes`.
                Ignored if no extra logit layer is appended.

            `"clas_strategy"`: str
                The classification strategy, one of:

                - **cls_time**: Sequence-level classification based on the
                  output of the first time step (which is the `CLS` token).
                  Each sequence has a class.
                - **all_time**: Sequence-level classification based on
                  the output of all time steps. Each sequence has a class.
                - **time_wise**: Step-wise classification, i.e., make
                  classification for each time step based on its output.

            `"max_seq_length"`: int, optional
                Maximum possible length of input sequences. Required if
                `clas_strategy` is `all_time`.

            `"dropout"`: float
                The dropout rate of the BERT encoder output.

            `"name"`: str
                Name of the classifier.
        """

        hparams = BERTEncoder.default_hparams()
        hparams.update({
            "num_classes": 2,
            "logit_layer_kwargs": None,
            "clas_strategy": "cls_time",
            "max_seq_length": None,
            "dropout": 0.1,
            "name": "bert_classifier"
        })
        return hparams

    def forward(self,  # type: ignore
                inputs: Union[torch.Tensor, torch.LongTensor],
                sequence_length: Optional[torch.LongTensor] = None,
                segment_ids: Optional[torch.LongTensor] = None) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        r"""Feeds the inputs through the network and makes classification.

        The arguments are the same as in
        :class:`~texar.torch.modules.BERTEncoder`.

        Args:
            inputs: Either a **2D Tensor** of shape `[batch_size, max_time]`,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time]`, containing the segment ids
                of tokens in input sequences. If `None` (default), a tensor
                with all elements set to zero is used.

        Returns:
            A tuple `(logits, preds)`, containing the logits over classes and
            the predictions, respectively.

            - If ``clas_strategy`` is ``cls_time`` or ``all_time``:

                - If ``num_classes`` == 1, ``logits`` and ``pred`` are both of
                  shape ``[batch_size]``.
                - If ``num_classes`` > 1, ``logits`` is of shape
                  ``[batch_size, num_classes]`` and ``pred`` is of shape
                  ``[batch_size]``.

            - If ``clas_strategy`` is ``time_wise``:

                - ``num_classes`` == 1, ``logits`` and ``pred`` are both of
                  shape ``[batch_size, max_time]``.
                - If ``num_classes`` > 1, ``logits`` is of shape
                  ``[batch_size, max_time, num_classes]`` and ``pred`` is of
                  shape ``[batch_size, max_time]``.
        """
        enc_outputs, pooled_output = self._encoder(inputs,
                                                   sequence_length,
                                                   segment_ids)
        # Compute logits
        strategy = self._hparams.clas_strategy
        if strategy == 'time_wise':
            logits = enc_outputs
        elif strategy == 'cls_time':
            logits = pooled_output
        elif strategy == 'all_time':
            # Pad `enc_outputs` to have max_seq_length before flatten
            length_diff = self._hparams.max_seq_length - inputs.shape[1]
            logit_input = F.pad(enc_outputs, [0, 0, 0, length_diff, 0, 0])
            logit_input_dim = (self._encoder.output_size *
                               self._hparams.max_seq_length)
            logits = logit_input.view(-1, logit_input_dim)
        else:
            raise ValueError('Unknown classification strategy: {}'.format(
                strategy))

        if self._logits_layer is not None:
            logits = self._dropout_layer(logits)
            logits = self._logits_layer(logits)

        # Compute predictions
        if strategy == "time_wise":
            if self.is_binary:
                logits = torch.squeeze(logits, -1)
                preds = (logits > 0).long()
            else:
                preds = torch.argmax(logits, dim=-1)
        else:
            if self.is_binary:
                preds = (logits > 0).long()
                logits = torch.flatten(logits)
            else:
                preds = torch.argmax(logits, dim=-1)
            preds = torch.flatten(preds)

        return logits, preds

    @property
    def output_size(self) -> int:
        r"""The feature size of :meth:`forward` output :attr:`logits`.
        If :attr:`logits` size is only determined by input
        (i.e. if ``num_classes`` == 1), the feature size is equal to ``-1``.
        Otherwise it is equal to last dimension value of :attr:`logits` size.
        """
        if self._hparams.num_classes == 1:
            logit_dim = -1
        elif self._hparams.num_classes > 1:
            logit_dim = self._hparams.num_classes
        elif self._hparams.clas_strategy == 'all_time':
            logit_dim = (self._encoder.output_size *
                         self._hparams.max_seq_length)
        elif self._hparams.clas_strategy == 'cls_time':
            logit_dim = self._encoder.output_size
        elif self._hparams.clas_strategy == 'time_wise':
            logit_dim = self._hparams.encoder.dim

        return logit_dim
