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
RoBERTa classifier.
"""
from typing import Optional, Tuple, Union

import torch

from texar.torch.modules.encoders.roberta_encoder import RoBERTaEncoder
from texar.torch.modules.classifiers.bert_classifier import BERTClassifier
from texar.torch.modules.pretrained.roberta import \
    PretrainedRoBERTaMixin

__all__ = [
    "RoBERTaClassifier"
]


class RoBERTaClassifier(PretrainedRoBERTaMixin, BERTClassifier):
    r"""Classifier based on RoBERTa modules. Please see
    :class:`~texar.torch.modules.PretrainedRoBERTaMixin` for a brief description
    of RoBERTa.

    This is a combination of the
    :class:`~texar.torch.modules.RoBERTaEncoder` with a classification
    layer. Both step-wise classification and sequence-level classification
    are supported, specified in :attr:`hparams`.

    Arguments are the same as in
    :class:`~texar.torch.modules.RoBERTaEncoder`.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``roberta-base``). Please refer to
            :class:`~texar.torch.modules.PretrainedRoBERTaMixin` for
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
    _ENCODER_CLASS = RoBERTaEncoder

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Same hyperparameters as in RoBertaEncoder
                ...
                # (2) Additional hyperparameters
                "num_classes": 2,
                "logit_layer_kwargs": None,
                "clas_strategy": "cls_time",
                "max_seq_length": None,
                "dropout": 0.1,
                "name": "roberta_classifier"
            }

        Here:

        1. Same hyperparameters as in
           :class:`~texar.torch.modules.RoBERTaEncoder`.
           See the :meth:`~texar.torch.modules.RoBERTaEncoder.default_hparams`.
           An instance of RoBERTaEncoder is created for feature extraction.

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
                The dropout rate of the RoBERTa encoder output.

            `"name"`: str
                Name of the classifier.
        """

        hparams = RoBERTaEncoder.default_hparams()
        hparams.update({
            "num_classes": 2,
            "logit_layer_kwargs": None,
            "clas_strategy": "cls_time",
            "max_seq_length": None,
            "dropout": 0.1,
            "name": "roberta_classifier"
        })
        return hparams

    def forward(self,  # type: ignore
                inputs: Union[torch.Tensor, torch.LongTensor],
                sequence_length: Optional[torch.LongTensor] = None) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        r"""Feeds the inputs through the network and makes classification.

        The arguments are the same as in
        :class:`~texar.torch.modules.RoBERTaEncoder`.

        Args:
            inputs: Either a **2D Tensor** of shape `[batch_size, max_time]`,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.

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
        logits, preds = super().forward(inputs=inputs,
                                        sequence_length=sequence_length,
                                        segment_ids=None)
        return logits, preds
