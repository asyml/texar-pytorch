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
Various RNN classifiers.
"""

from typing import Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from torch.nn import functional as F

from texar.torch.core.cell_wrappers import RNNCellBase
from texar.torch.hyperparams import HParams
from texar.torch.modules.classifiers.classifier_base import ClassifierBase
from texar.torch.modules.encoders.rnn_encoders import \
        UnidirectionalRNNEncoder
from texar.torch.utils.utils import dict_fetch


__all__ = [
    "UnidirectionalRNNClassifier",
]

State = TypeVar('State')


class UnidirectionalRNNClassifier(ClassifierBase):
    r"""One directional RNN classifier. This is a combination of the
    :class:`~texar.torch.modules.UnidirectionalRNNEncoder` with a classification
    layer. Both step-wise classification and sequence-level classification
    are supported, specified in :attr:`hparams`.

    Arguments are the same as in
    :class:`~texar.torch.modules.UnidirectionalRNNEncoder`.

    Args:
        input_size (int): The number of expected features in the input for the
            cell.
        cell: (RNNCell, optional) If not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        output_layer (optional): An instance of
            :torch_nn:`Module`. Applies to the RNN cell
            output of each step. If `None` (default), the output layer is
            created as specified in :attr:`hparams["output_layer"]`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.
    """

    def __init__(self,
                 input_size: int,
                 cell: Optional[RNNCellBase[State]] = None,
                 output_layer: Optional[nn.Module] = None,
                 hparams=None):

        super().__init__(hparams=hparams)

        # Create the underlying encoder
        encoder_hparams = dict_fetch(
            hparams, UnidirectionalRNNEncoder.default_hparams())

        self._encoder = UnidirectionalRNNEncoder(
            input_size=input_size,
            cell=cell,
            output_layer=output_layer,
            hparams=encoder_hparams)

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

        self.is_binary = (self.num_classes == 1) or \
                         (self.num_classes <= 0 and
                          self._encoder.output_size == 1)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Same hyperparameters as in UnidirectionalRNNEncoder
                ...

                # (2) Additional hyperparameters
                "num_classes": 2,
                "logit_layer_kwargs": None,
                "clas_strategy": "final_time",
                "max_seq_length": None,
                "name": "unidirectional_rnn_classifier"
            }

        Here:

        1. Same hyperparameters as in
           :class:`~texar.torch.modules.UnidirectionalRNNEncoder`.
           See the
           :meth:`~texar.torch.modules.UnidirectionalRNNEncoder.default_hparams`
           . An instance of UnidirectionalRNNEncoder is created for feature
           extraction.

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

                - **final_time**: Sequence-level classification based on the
                  output of the final time step. Each sequence has a class.
                - **all_time**: Sequence-level classification based on
                  the output of all time steps. Each sequence has a class.
                - **time_wise**: Step-wise classification, i.e., make
                  classification for each time step based on its output.

            `"max_seq_length"`: int, optional
                Maximum possible length of input sequences. Required if
                `clas_strategy` is `all_time`.

            `"name"`: str
                Name of the classifier.
        """
        hparams = UnidirectionalRNNEncoder.default_hparams()
        hparams.update({
            "num_classes": 2,
            "logit_layer_kwargs": None,
            "clas_strategy": "final_time",
            "max_seq_length": None,
            "name": "bert_classifier"
        })
        return hparams

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                sequence_length: Optional[torch.LongTensor] = None,
                initial_state: Optional[State] = None,
                time_major: bool = False) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        r"""Feeds the inputs through the network and makes classification.

        The arguments are the same as in
        :class:`~texar.torch.modules.UnidirectionalRNNEncoder`.

        Args:
            inputs: A 3D Tensor of shape ``[batch_size, max_time, dim]``.
                The first two dimensions
                :attr:`batch_size` and :attr:`max_time` are exchanged if
                :attr:`time_major` is `True`.
            sequence_length (optional): A 1D :tensor:`LongTensor` of shape
                ``[batch_size]``.
                Sequence lengths of the batch inputs. Used to copy-through
                state and zero-out outputs when past a batch element's sequence
                length.
            initial_state (optional): Initial state of the RNN.
            time_major (bool): The shape format of the :attr:`inputs` and
                :attr:`outputs` Tensors. If `True`, these tensors are of shape
                ``[max_time, batch_size, depth]``. If `False` (default),
                these tensors are of shape ``[batch_size, max_time, depth]``.

        Returns:
            A tuple `(logits, preds)`, containing the logits over classes and
            the predictions, respectively.

            - If ``clas_strategy`` is ``final_time`` or ``all_time``:

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
                - If ``time_major`` is `True`, the batch and time dimensions
                  are exchanged.
        """
        enc_outputs, _ = self._encoder(inputs=inputs,
                                       sequence_length=sequence_length,
                                       initial_state=initial_state,
                                       time_major=time_major)
        # Compute logits
        strategy = self._hparams.clas_strategy
        if strategy == 'time_wise':
            logits = enc_outputs
        elif strategy == 'final_time':
            if time_major:
                logits = enc_outputs[-1, :, :]
            else:
                logits = enc_outputs[:, -1, :]
        elif strategy == 'all_time':
            if time_major:
                length_diff = self._hparams.max_seq_length - inputs.shape[0]
                logit_input = F.pad(enc_outputs, [0, length_diff, 0, 0, 0, 0])
                logit_input_dim = (self._encoder.output_size *
                                   self._hparams.max_seq_length)
                logits = logit_input.view(-1, logit_input_dim)
            else:
                length_diff = self._hparams.max_seq_length - inputs.shape[1]
                logit_input = F.pad(enc_outputs, [0, 0, 0, length_diff, 0, 0])
                logit_input_dim = (self._encoder.output_size *
                                   self._hparams.max_seq_length)
                logits = logit_input.view(-1, logit_input_dim)
        else:
            raise ValueError('Unknown classification strategy: {}'.format(
                strategy))

        if self._logits_layer is not None:
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
        elif self._hparams.clas_strategy == 'final_time':
            logit_dim = self._encoder.output_size
        elif self._hparams.clas_strategy == 'time_wise':
            logit_dim = self._hparams.encoder.dim

        return logit_dim
