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
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import texar as tx
from texar.hyperparams import HParams
from texar.modules.classifiers.classifier_base import ClassifierBase

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
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            `default_hparams` for the hyper-parameter structure
            and default values.

    .. document private functions
    """

    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.word_embedder = tx.modules.WordEmbedder(
            vocab_size=self._hparams.vocab_size, hparams=self._hparams.embed)

        # Segment embedding for each type of tokens
        self.segment_embedder = tx.modules.WordEmbedder(
            vocab_size=self._hparams.type_vocab_size,
            hparams=self._hparams.segment_embed)

        # Position embedding
        self.position_embedder = tx.modules.PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed)

        # The BERT encoder (a TransformerEncoder)
        self.encoder = tx.modules.TransformerEncoder(
            hparams=self._hparams.encoder)

        self.pooler = nn.Sequential(
            nn.Linear(self._hparams.hidden_size, self._hparams.hidden_size),
            nn.Tanh(),
            nn.Dropout(self._hparams.dropout))

        self._num_classes = self._hparams.num_classes

        if self._num_classes > 0:
            logit_kwargs = self._hparams.logit_layer_kwargs
            if logit_kwargs is None:
                logit_kwargs = {}
            elif not isinstance(logit_kwargs, HParams):
                raise ValueError("hparams['logit_layer_kwargs'] "
                                 "must be a dict.")
            else:
                logit_kwargs = logit_kwargs.todict()

            self.logits_layer = nn.Linear(
                self._hparams.hidden_size, self._num_classes, **logit_kwargs)

        else:
            self.logits_layer = None

        self.step_iteration = 0

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python
            {
                # (1) Different modules for sentence encoder
                # (2) Additional hyperparameters
                "num_classes": 2,
                "logit_layer_kwargs": None,
                "clas_strategy": "cls_time",
                "max_seq_length": None,
                "dropout": 0.1,
                "name": "bert_classifier"
            }

        Here:
        1. Three parts of embedders. Two for word level tokens, segment IDs
        where each has the same hyperparameters as in
        :class:`~texar.modules.embedders.WordEmbedder`. The third is position
        embedder which is used for
        :class:`~texar.modules.embedders.PositionEmbedder`

        2. Additional hyperparameters for downstream tasks:

            "num_classes" : int
                Number of classes:

                - If **`> 0`**, an additional :torch_nn:`Linear` layer is
                  appended to the encoder to compute the logits over classes.
                - If **`<= 0`**, no dense layer is appended. The number of
                  classes is assumed to be the final dense layer size of the
                  encoder.

            "logit_layer_kwargs" : dict
                Keyword arguments for the logit Dense layer constructor,
                except for argument "units" which is set to "num_classes".
                Ignored if no extra logit layer is appended.

            "clas_strategy" : str
                The classification strategy, one of:
                - **"cls_time"**: Sequence-level classification based on the
                  output of the first time step (which is the "CLS" token).
                  Each sequence has a class.
                - **"all_time"**: Sequence-level classification based on
                  the output of all time steps. Each sequence has a class.
                - **"time_wise"**: Step-wise classification, i.e., make
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
            "segment_embed": {
                "dim": _default_input_dim,
                "name": "token_type_embeddings",
            },
            "type_vocab_size": 2,
            "position_embed": {
                "dim": _default_input_dim,
                "name": "position_embeddings",
            },
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
                            },
                        },
                        {"type": "GELU"},
                        {
                            "type": "Linear",
                            "kwargs": {
                                "in_features": _default_input_dim * 4,
                                "out_features": _default_input_dim,
                                "bias": True,
                            },
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

    def forward(self, inputs: torch.Tensor,  # type: ignore
                sequence_length: Optional[torch.LongTensor] = None,
                segment_ids: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None) \
            -> Tuple[torch.Tensor, torch.LongTensor, Optional[torch.Tensor]]:
        """Feeds the inputs through the network and makes classification.

        The arguments are the same as in
        :class:`~texar.modules.BertEncoder`.

        Args:
            inputs: A 2D Tensor of shape ``[batch_size, max_time]``,
                containing the token ids of tokens in input sequences.
            sequence_length (optional): A 1D Tensor of shape ``[batch_size]``.
                Input tokens beyond respective sequence lengths are masked
                out automatically.
            segment_ids (optional): A 2D Tensor of shape
                ``[batch_size, max_time]``, containing the segment ids
                of tokens in input sequences. If ``None`` (default), a tensor
                with all elements set to zero is used.
            labels (optional): Used in training

        Returns:
            A tuple ``(logits, pred, loss)``, containing the logits over
            classes, the predictions, and the final loss, respectively. If
            ``labels`` is ``None``, then the returned loss is also ``None``.

            - If ``"clas_strategy"`` is ``"cls_time"`` or ``"all_time"``:

                - If ``"num_classes"``==1, ``logits`` and ``pred`` are both of
                  shape ``[batch_size]``.
                - If ``"num_classes"``>1, `logits` is of shape
                  ``[batch_size, num_classes]`` and ``pred`` is of shape
                  `[batch_size]`.

            - If ``"clas_strategy"`` is ``"time_wise"``:

                - If ``"num_classes"``==1, ``logits`` and ``pred`` are both of
                  shape ``[batch_size, max_time]``.
                - If ``"num_classes"``>1, ``logits`` is of shape
                  ``[batch_size, max_time, num_classes]`` and ``pred`` is of
                  shape ``[batch_size, max_time]``.
        """
        word_embeds = self.word_embedder(inputs)
        segment_embeds = self.segment_embedder(segment_ids)
        seq_length = torch.full(
            (inputs.size()[0],), inputs.size()[1], dtype=torch.int32)
        seq_length = seq_length.to(device=inputs.device)
        pos_embeds = self.position_embedder(sequence_length=seq_length)
        input_embeds = word_embeds + segment_embeds + pos_embeds

        enc_output = self.encoder(input_embeds, sequence_length)

        # Compute logits
        strategy = self._hparams.clas_strategy
        if strategy == "time_wise":
            logits_input = enc_output
        elif strategy == "cls_time":
            first_token_tensor = enc_output[:, 0:1, :].squeeze()
            pooled_output = self.pooler(first_token_tensor)
            logits_input = pooled_output
        elif strategy == "all_time":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown classification strategy: {strategy}")

        if self.logits_layer is not None:
            logits = self.logits_layer(logits_input)
        else:
            logits = logits_input

        num_classes = self._num_classes
        is_binary = num_classes == 1
        is_binary = is_binary or (num_classes <= 0 and logits.shape[-1] == 1)

        if strategy == "time_wise":
            if is_binary:
                logits = torch.squeeze(logits, -1)
                preds = (logits > 0).long()
            else:
                preds = torch.argmax(logits, dim=-1)
        else:
            if is_binary:
                preds = (logits > 0).long()
                logits = torch.flatten(logits)
            else:
                preds = torch.argmax(logits, dim=-1)
            preds = torch.flatten(preds)

        if labels is not None:
            if is_binary:
                loss = F.binary_cross_entropy(
                    logits.view(-1), labels.view(-1), reduction='mean')
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self._num_classes), labels.view(-1),
                    reduction='mean')
        else:
            loss = None

        return logits, preds, loss
