import math
from typing import Callable
import pickle

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
BERT classifiers.
"""

import texar as tx
from texar.modules.classifiers.classifier_base import ClassifierBase
from texar.hyperparams import HParams

import torch
from torch import nn
from torch.nn import functional as F

# pylint: disable=too-many-arguments, invalid-name, no-member,
# pylint: disable=too-many-branches, too-many-locals, too-many-statements

__all__ = ["BertClassifier"]


class BertClassifier(ClassifierBase):
    """Classifier based on bert modules.

    This is a combination of the
    :class:`~texar.modules.BertEncoder` with a classification
    layer. Both step-wise classification and sequence-level classification
    are supported, specified in :attr:`hparams`.

    Arguments are the same as in
    :class:`~texar.modules.BertEncoder`.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model to load selected in the list of:
            `bert-base-uncased`, `bert-large-uncased`, `bert-base-cased`,
            `bert-large-cased`, `bert-base-multilingual-uncased`,
            `bert-base-multilingual-cased`, `bert-base-chinese`.
            If `None`, will use the model name in :attr:`hparams`.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture
            and default values.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, hparams=None):

        ClassifierBase.__init__(self, hparams)
        self.word_embedder = tx.modules.WordEmbedder(
            vocab_size=self._hparams.vocab_size, hparams=self._hparams.embed
        )

        # Segment embedding for each type of tokens
        self.segment_embedder = tx.modules.WordEmbedder(
            vocab_size=self._hparams.type_vocab_size,
            hparams=self._hparams.segment_embed,
        )

        # Position embedding
        self.position_embedder = tx.modules.PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed,
        )

        # The BERT encoder (a TransformerEncoder)
        self.encoder = tx.modules.TransformerEncoder(
            hparams=self._hparams.encoder)

        self.pooler = nn.ModuleList(
            [
                nn.Linear(self._hparams.hidden_size, self._hparams.hidden_size),
                nn.Dropout(self._hparams.dropout),
            ]
        )

        self._num_classes = self._hparams.num_classes

        if self._num_classes > 0:
            logit_kwargs = self._hparams.logit_layer_kwargs
            if logit_kwargs is None:
                logit_kwargs = {}
            elif not isinstance(logit_kwargs, HParams):
                raise ValueError(
                    "hparams['logit_layer_kwargs'] must be a dict."
                )
            else:
                logit_kwargs = logit_kwargs.todict()
            logit_kwargs.update({
                "in_features": self._hparams.hidden_size,
                "out_features": self._num_classes,
                "bias": True})

            self.logits_layer = nn.Linear(**logit_kwargs)

        self.step_iteration = 0

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

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
        :class:`~texar.modules.BertEncoder`.
        See the :meth:`~texar.modules.BertEncoder.default_hparams`.
        An instance of BertEncoder is created for feature extraction.

        2. Additional hyperparameters:

            "num_classes" : int
                Number of classes:

                - If **`> 0`**, an additional :tf_main:`Dense <layers/Dense>` \
                layer is appended to the encoder to compute the logits over \
                classes.
                - If **`<= 0`**, no dense layer is appended. The number of \
                classes is assumed to be the final dense layer size of the \
                encoder.

            "logit_layer_kwargs" : dict
                Keyword arguments for the logit Dense layer constructor,
                except for argument "units" which is set to "num_classes".
                Ignored if no extra logit layer is appended.

            "clas_strategy" : str
                The classification strategy, one of:
                - **"cls_time"**: Sequence-level classification based on the \
                output of the first time step (which is the "CLS" token). \
                Each sequence has a class.
                - **"all_time"**: Sequence-level classification based on \
                the output of all time steps. Each sequence has a class.
                - **"time_wise"**: Step-wise classfication, i.e., make \
                classification for each time step based on its output.

            "max_seq_length" : int, optional
                Maximum possible length of input sequences. Required if
                "clas_strategy" is "all_time".

            "dropout" : float
                The dropout rate of the bert encoder output.

            "name" : str
                Name of the classifier.
        """

        _default_input_dim = 768
        hparams = {
            "embed": {"dim": _default_input_dim, "name": "word_embeddings"},
            "vocab_size": 30522,
            "segment_embed": {"dim": _default_input_dim, "name": "token_type_embeddings"},
            "type_vocab_size": 2,
            "position_embed": {"dim": _default_input_dim, "name": "position_embeddings"},
            "position_size": 512,
            "encoder": {
                "dim": _default_input_dim,
                "embedding_dropout": 0.1,
                "multihead_attention": {
                    "dropout_rate": 0.1,
                    "name": "self",
                    "num_heads": 12,
                    "num_units": _default_input_dim,
                    "output_dim": _default_input_dim,
                    "use_bias": True,
                },
                "name": "encoder",
                "num_blocks": 12,
                "poswise_feedforward": {
                    "layers": [
                        {
                            "type": "Linear",
                            "kwargs": {
                                "in_features": _default_input_dim,
                                "out_features": _default_input_dim * 4,
                                "bias": True,
                            }
                        },
                        {
                            "type": "GELU",
                        },
                        {
                            "type": "Linear",
                            "kwargs": {
                                "in_features": _default_input_dim * 4,
                                "out_features": _default_input_dim,
                                "bias": True,
                            }
                        },
                    ]
                },
                "residual_dropout": 0.1,
                "use_bert_config": True,
            },
            "hidden_size": _default_input_dim,
            "initializer": None,
            "num_classes": 2,
            "logit_layer_kwargs": None,
            "clas_strategy": "cls_time",
            "max_seq_length": None,
            "dropout": 0.1,
            "name": "bert_classifier",
        }

        return hparams

    def forward(self, inputs, sequence_length=None, segment_ids=None,
                labels=None, **kwargs):
        """Feeds the inputs through the network and makes classification.

        The arguments are the same as in
        :class:`~texar.modules.BertEncoder`.

        Args:
            inputs: A 2D Tensor of shape `[batch_size, max_time]`,
                containing the token ids of tokens in input sequences.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time]`, containing the segment ids
                of tokens in input sequences. If `None` (default), a tensor
                with all elements set to zero is used.
            labels (optional): Used in training
            **kwargs: Keyword arguments.

        Returns:
            A tuple `(logits, pred)`, containing the logits over classes and
            the predictions, respectively.

            - If "clas_strategy"=="cls_time" or "all_time"

                - If "num_classes"==1, `logits` and `pred` are of both \
                shape `[batch_size]`
                - If "num_classes">1, `logits` is of shape \
                `[batch_size, num_classes]` and `pred` is of shape \
                `[batch_size]`.

            - If "clas_strategy"=="time_wise",

                - If "num_classes"==1, `logits` and `pred` are of both \
                shape `[batch_size, max_time]`
                - If "num_classes">1, `logits` is of shape \
                `[batch_size, max_time, num_classes]` and `pred` is of shape \
                `[batch_size, max_time]`.
        """

        word_embeds = self.word_embedder(inputs)
        segment_embeds = self.segment_embedder(segment_ids)
        seq_length = torch.full((inputs.size()[0]), inputs.size()[1],
                                dtype=torch.int32)
        pos_embeds = self.position_embedder(
            sequence_length=seq_length)
        input_embeds = word_embeds + segment_embeds + pos_embeds
        enc_output = self.encoder(input_embeds, sequence_length)
        pooled_output = self.pooler(enc_output)

        # Compute logits
        stra = self._hparams.clas_strategy
        if stra == "time_wise":
            logits_input = enc_output
        elif stra == "cls_time":
            logits_input = pooled_output
        elif stra == "all_time":
            raise NotImplementedError
        else:
            raise ValueError("Unknown classification strategy: {}".format(stra))

        if getattr(self, 'logit_layer', None):
            logits = self.logit_layer(logits_input)
        else:
            logits = logits_input

        num_classes = self._hparams.num_classes
        is_binary = num_classes == 1
        is_binary = is_binary or (num_classes <= 0 and logits.shape[-1] == 1)

        if stra == "time_wise":
            if is_binary:
                logits = torch.squeeze(logits, -1)
                preds = (logits > 0).int()
            else:
                preds = torch.argmax(logits, dim=-1)
        else:
            if is_binary:
                preds = (logits > 0).int()
                logits = torch.reshape(logits, [-1])
            else:
                preds = torch.argmax(logits, dim=-1)
            preds = torch.reshape(preds, [-1])

        if labels is not None:
            loss = torch.sum(-labels * F.log_softmax(logits, -1), -1).mean()
        else:
            loss = None

        return logits, preds, loss
