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
Various position embedders.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from texar.torch.modules.embedders import embedder_utils
from texar.torch.modules.embedders.embedder_base import (
    EmbedderBase, EmbeddingDropout)
from texar.torch.utils.shapes import mask_sequences

__all__ = [
    "PositionEmbedder",
    "SinusoidsPositionEmbedder",
]


class PositionEmbedder(EmbedderBase):
    r"""Simple position embedder that maps position indexes into embeddings
    via lookup.

    Either :attr:`init_value` or :attr:`position_size` is required. If both are
    given, there must be ``init_value.shape[0]==position_size``.

    Args:
        init_value (optional): A Tensor or numpy array that contains the
            initial value of embeddings. It is typically of shape
            ``[position_size, embedding dim]``.

            If `None`, embedding is initialized as specified in
            ``hparams["initializer"]``. Otherwise, the
            ``"initializer"`` and ``"dim"``
            hyperparameters in :attr:`hparams` are ignored.
        position_size (int, optional): The number of possible positions, e.g.,
            the maximum sequence length. Required if :attr:`init_value` is
            not given.
        hparams (dict, optional): Embedder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the structure and default values.


    .. document private functions
    """

    def __init__(self, position_size: Optional[int] = None,
                 init_value: Optional[torch.Tensor] = None, hparams=None):

        if init_value is None and position_size is None:
            raise ValueError(
                "Either `init_value` or `position_size` is required.")

        super().__init__(position_size, init_value, hparams=hparams)

        self._position_size = position_size
        if position_size is None:
            self._position_size = self._num_embeds
        if self._position_size != self._num_embeds:
            raise ValueError(
                f"position_size must be equal to init_value.shape[0]. "
                f"Got {self._position_size} and {self._num_embeds}")

        self._built = True
        self._dropout_layer = EmbeddingDropout(self._hparams.dropout_rate)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "dim": 100,
                "initializer": {
                    "type": "random_uniform_initializer",
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                },
                "dropout_rate": 0,
                "name": "position_embedder"
            }

        The hyperparameters have the same meaning as those in
        :meth:`texar.torch.modules.WordEmbedder.default_hparams`.
        """
        hparams = embedder_utils.default_embedding_hparams()
        hparams["name"] = "position_embedder"
        return hparams

    def extra_repr(self) -> str:
        return f"position_size={self.position_size}, embedding_dim={self.dim}"

    def forward(self,  # type: ignore
                positions: Optional[torch.LongTensor] = None,
                sequence_length: Optional[torch.LongTensor] = None, **kwargs):
        r"""Embeds the positions.

        Either :attr:`positions` or :attr:`sequence_length` is required:

            - If both are given, :attr:`sequence_length` is used to mask out
              embeddings of those time steps beyond the respective sequence
              lengths.
            - If only :attr:`sequence_length` is given, then positions
              from 0 to ``sequence_length - 1`` are embedded.

        Args:
            positions (optional): A :tensor:`LongTensor` containing the position
                IDs to embed.
            sequence_length (optional): An :tensor:`LongTensor` of shape
                ``[batch_size]``. Time steps beyond the respective sequence
                lengths will have zero-valued embeddings.
            kwargs: Additional keyword arguments for
                :torch_nn:`functional.embedding` besides
                :attr:`params` and :attr:`ids`.

        Returns:
            A `Tensor` of shape `shape(inputs) + embedding dimension`.
        """
        # Gets embedder inputs
        if positions is None:
            if sequence_length is None:
                raise ValueError(
                    'Either `positions` or `sequence_length` is required.')
            max_length = torch.max(sequence_length)
            single_inputs = torch.arange(start=0, end=max_length)
            # Expands `single_inputs` to have shape [batch_size, max_length]
            inputs = single_inputs.unsqueeze(0)
            inputs = inputs.expand(len(sequence_length), -1).contiguous()
        else:
            inputs = positions

        ids_rank = inputs.dim()
        embedding = self._embedding
        inputs = inputs.to(device=embedding.device)
        # Gets dropout strategy
        st = self._hparams.dropout_strategy

        # Dropouts as 'item_type' before embedding
        if st == 'item_type':
            noise_shape = self._get_noise_shape(
                dropout_strategy=st, dropout_input=embedding)
            embedding = self._dropout_layer(embedding, noise_shape)

        # Embeds
        outputs = torch.nn.functional.embedding(
            inputs.type(torch.long), embedding, **kwargs)

        # Dropouts as 'item' or 'elements' after embedding
        if st != 'item_type':
            noise_shape = self._get_noise_shape(
                dropout_strategy=st, dropout_input=outputs, ids_rank=ids_rank)
            outputs = self._dropout_layer(outputs, noise_shape)

        # Optionally masks
        if sequence_length is not None:
            outputs = mask_sequences(
                outputs, sequence_length)

        return outputs

    @property
    def embedding(self):
        r"""The embedding tensor.
        """
        return self._embedding

    @property
    def dim(self):
        r"""The embedding dimension.
        """
        return self._dim

    @property
    def position_size(self):
        r"""The position size, i.e., maximum number of positions.
        """
        return self._position_size

    @property
    def output_size(self) -> int:
        r"""The feature size of :meth:`forward` output. If the :attr:`dim`
        hyperparameter is a ``list`` or ``tuple``, the feature size
        equals its final dimension; otherwise, if :attr:`dim` is an
        ``int``, the feature size equals :attr:`dim`.
        """
        if isinstance(self._dim, (list, tuple)):
            return self._dim[-1]
        else:
            return self._dim


class SinusoidsPositionEmbedder(EmbedderBase):
    r"""Sinusoid position embedder that maps position indexes into embeddings
    via sinusoid calculation. This module does not have trainable parameters.
    Used in, e.g., Transformer models
    `(Vaswani et al.) "Attention Is All You Need"`.

    Each channel of the input Tensor is incremented by a sinusoid of a
    different frequency and phase.
    This allows attention to learn to use absolute and relative positions.

    Timing signals should be added to some precursors of both the query
    and the memory inputs to attention.
    The use of relative position is possible because `sin(x+y)` and
    `cos(x+y)` can be expressed in terms of `y`, `sin(x)`, and `cos(x)`.
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to ``dim / 2``. For each timescale, we
    generate the two sinusoidal signals `sin(timestep/timescale)` and
    `cos(timestep/timescale)`.  All of these sinusoids are concatenated in
    the dim dimension.

    Args:
        position_size (int): The number of possible positions, e.g., the maximum
            sequence length. Set ``position_size=None`` and
            ``hparams['cache_embeddings']=False`` to use arbitrarily large or
            negative position indices.

    .. document private functions
    """
    signal: torch.Tensor
    inv_timescales: torch.Tensor

    def __init__(self, position_size: Optional[int] = None, hparams=None):
        super().__init__(hparams=hparams)
        self._num_embeds = position_size  # type: ignore
        self._dim = self._hparams.dim
        self._cache_embeddings = self._hparams.cache_embeddings

        num_timescales = self._dim // 2
        min_timescale = self._hparams.min_timescale
        max_timescale = self._hparams.max_timescale

        log_timescale_increment = (math.log(max_timescale / min_timescale) /
                                   (num_timescales - 1))
        inv_timescales = min_timescale * torch.exp(
            (torch.arange(num_timescales, dtype=torch.float) *
             -log_timescale_increment))
        if self._cache_embeddings:
            if position_size is None:
                raise ValueError("'position_size' must not be None when "
                                 "'cache_embeddings' is set to True")
            positions = torch.arange(position_size, dtype=torch.float)
            signal = self._compute_embeddings(positions, inv_timescales)
            self.register_buffer('signal', signal)
        else:
            self.register_buffer('inv_timescales', inv_timescales)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values
        We use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale. The number of different
        timescales is equal to ``dim / 2``.

        .. code-block:: python

            {
                'min_timescale': 1.0,
                'max_timescale': 10000.0,
                'dim': 512,
                'cache_embeddings': True,
                'name':'sinusoid_position_embedder',
            }

        Here:

        `"cache_embeddings"`: bool
            If `True`, precompute embeddings for positions in range
            `[0, position_size - 1]`. This leads to faster lookup but requires
            lookup indices to be within this range.

            If `False`, embeddings are computed on-the-fly during lookup. Set to
            `False` if your application needs to handle sequences of arbitrary
            length, or requires embeddings at negative positions.
        """
        return {
            'min_timescale': 1.0,
            'max_timescale': 1.0e4,
            'dim': 512,
            'cache_embeddings': True,
            'name': 'sinusoid_position_embedder',
        }

    def extra_repr(self) -> str:
        return f"embedding_dim={self.dim}"

    def _compute_embeddings(self, positions: torch.Tensor,
                            inv_timescales: torch.Tensor) -> torch.Tensor:
        scaled_time = (positions.type_as(inv_timescales).view(-1, 1) *
                       inv_timescales.unsqueeze(0))
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        if self._dim % 2 == 1:
            # An extra dimension must be added when dimension is odd.
            signal = torch.cat(
                [signal, signal.new_zeros(signal.size(0), 1)], dim=1)
        signal = signal.view(*positions.size(), -1).contiguous()
        return signal

    def forward(self,  # type: ignore
                positions: Optional[torch.LongTensor] = None,
                sequence_length: Optional[torch.LongTensor] = None, **kwargs) \
            -> torch.Tensor:
        r"""Embeds.
        Either :attr:`positions` or :attr:`sequence_length` is required:

        - If both are given, :attr:`sequence_length` is used to mask out
          embeddings of those time steps beyond the respective sequence
          lengths.
        - If only :attr:`sequence_length` is given, then positions
          from `0` to `sequence_length - 1` are embedded.

        Args:
            positions (optional): An :tensor:`LongTensor` containing the
                position IDs to embed.
            sequence_length (optional): An :tensor:`LongTensor` of shape
                ``[batch_size]``. Time steps beyond
                the respective sequence lengths will have zero-valued
                embeddings.

        Returns:
            A Tensor of shape ``[batch_size, position_size, dim]``.
        """
        if positions is None:
            if sequence_length is None:
                raise ValueError(
                    'Either `positions` or `sequence_length` is required.')
            max_length = sequence_length.max()
            batch_size = sequence_length.size(0)
            inputs = torch.arange(max_length).to(device=sequence_length.device)
            inputs = inputs.expand(batch_size, max_length)
        else:
            inputs = positions

        if self._cache_embeddings:
            outputs = F.embedding(inputs, self.signal, **kwargs)
        else:
            outputs = self._compute_embeddings(inputs, self.inv_timescales)

        if sequence_length is not None:
            outputs = mask_sequences(outputs, sequence_length)

        return outputs

    @property
    def dim(self):
        r"""The embedding dimension.
        """
        return self._dim

    @property
    def output_size(self) -> int:
        r"""The feature size of :meth:`forward` output. If the :attr:`dim`
        hyperparameter is a ``list`` or ``tuple``, the feature size
        equals its final dimension; otherwise, if :attr:`dim` is an
        ``int``, the feature size equals :attr:`dim`.
        """
        if isinstance(self._dim, (list, tuple)):
            dim = self._dim[-1]
        else:
            dim = self._dim
        return dim
