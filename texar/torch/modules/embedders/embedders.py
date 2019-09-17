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
Various embedders.
"""
from typing import Optional

import torch
from torch.nn import functional as F

from texar.torch.modules.embedders import embedder_utils
from texar.torch.modules.embedders.embedder_base import (
    EmbedderBase, EmbeddingDropout)

__all__ = [
    "WordEmbedder",
]


class WordEmbedder(EmbedderBase):
    r"""Simple word embedder that maps indexes into embeddings. The indexes
    can be soft (e.g., distributions over vocabulary).

    Either :attr:`init_value` or :attr:`vocab_size` is required. If both are
    given, there must be ``init_value.shape[0]==vocab_size``.

    Args:
        init_value (optional): A Tensor or numpy array that contains the
            initial value of embeddings. It is typically of shape
            ``[vocab_size] + embedding-dim``. Embeddings can have dimensionality
            > 1.

            If `None`, embedding is initialized as specified in
            ``hparams["initializer"]``. Otherwise, the
            ``"initializer"`` and ``"dim"`` hyperparameters in :attr:`hparams`
            are ignored.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`init_value` is not given.
        hparams (dict, optional): Embedder hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    See :meth:`forward` for the inputs and outputs of the embedder.

    Example:

    .. code-block:: python

        ids = torch.empty([32, 10]).uniform_(to=10).type(torch.int64).
        soft_ids = torch.empty([32, 10, 100]).uniform_()

        embedder = WordEmbedder(vocab_size=100, hparams={'dim': 256})
        ids_emb = embedder(ids=ids) # shape: [32, 10, 256]
        soft_ids_emb = embedder(soft_ids=soft_ids) # shape: [32, 10, 256]

    .. code-block:: python

        # Use with Texar data module
        hparams={
            'dataset': {
                'embedding_init': {'file': 'word2vec.txt'}
                ...
            },
        }
        data = MonoTextData(data_params)
        iterator = DataIterator(data)
        batch = next(iter(iterator))

        # Use data vocab size
        embedder_1 = WordEmbedder(vocab_size=data.vocab.size)
        emb_1 = embedder_1(batch['text_ids'])

        # Use pre-trained embedding
        embedder_2 = WordEmbedder(init_value=data.embedding_init_value)
        emb_2 = embedder_2(batch['text_ids'])


    .. document private functions
    """

    def __init__(self, init_value: Optional[torch.Tensor] = None,
                 vocab_size: Optional[int] = None, hparams=None):

        if init_value is None and vocab_size is None:
            raise ValueError(
                "Either `init_value` or `vocab_size` is required.")

        super().__init__(init_value=init_value,
                         num_embeds=vocab_size, hparams=hparams)

        if vocab_size is None:
            self._vocab_size = self._num_embeds
        else:
            self._vocab_size = vocab_size
        if self._vocab_size != self._num_embeds:
            raise ValueError(
                f"vocab_size must equal to init_value.shape[0]. "
                f"Got {self._vocab_size} and {self._num_embeds}")

        self._dropout_layer = EmbeddingDropout(self._hparams.dropout_rate)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "dim": 100,
                "dropout_rate": 0,
                "dropout_strategy": 'element',
                "initializer": {
                    "type": "random_uniform_initializer",
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                },
                "name": "word_embedder",
            }

        Here:

        `"dim"`: int or list
            Embedding dimension. Can be a list of integers to yield embeddings
            with dimensionality > 1.

            Ignored if :attr:`init_value` is given to the embedder constructor.

        `"dropout_rate"`: float
            The dropout rate between 0 and 1. For example, ``dropout_rate=0.1``
            would zero out 10% of the embeddings. Set to 0 to disable dropout.

        `"dropout_strategy"`: str
            The dropout strategy. Can be one of the following

            - ``"element"``: The regular strategy that drops individual elements
              in the embedding vectors.
            - ``"item"``: Drops individual items (e.g., words) entirely. For
              example, for the word sequence "the simpler the better", the
              strategy can yield "_ simpler the better", where the first "the"
              is dropped.
            - ``"item_type"``: Drops item types (e.g., word types). For example,
              for the above sequence, the strategy can yield "_ simpler _
              better", where the word type "the" is dropped. The dropout will
              never yield "_ simpler the better" as in the ``"item"`` strategy.

        `"initializer"`: dict or None
            Hyperparameters of the initializer for embedding values. See
            :func:`~texar.torch.core.get_initializer` for the details. Ignored
            if :attr:`init_value` is given to the embedder constructor.

        `"name"`: str
            Name of the embedding variable.
        """
        hparams = embedder_utils.default_embedding_hparams()
        hparams["name"] = "word_embedder"
        return hparams

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, embedding_dim={self.dim}"

    def forward(self,  # type: ignore
                ids: Optional[torch.LongTensor] = None,
                soft_ids: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        r"""Embeds (soft) ids.

        Either :attr:`ids` or :attr:`soft_ids` must be given, and they
        must not be given at the same time.

        Args:
            ids (optional): An integer tensor containing the ids to embed.
            soft_ids (optional): A tensor of weights (probabilities) used to
                mix the embedding vectors.
            kwargs: Additional keyword arguments for
                :torch_nn:`functional.embedding` besides :attr:`params` and
                :attr:`ids`.

        Returns:
            If :attr:`ids` is given, returns a Tensor of shape
            ``list(ids.shape) + embedding-dim``. For example,
            if ``list(ids.shape) == [batch_size, max_time]``
            and ``list(embedding.shape) == [vocab_size, emb_dim]``, then the
            return tensor has shape ``[batch_size, max_time, emb_dim]``.

            If :attr:`soft_ids` is given, returns a Tensor of shape
            ``list(soft_ids.shape)[:-1] + embedding-dim``. For example,
            if ``list(soft_ids.shape) == [batch_size, max_time, vocab_size]``
            and ``list(embedding.shape) == [vocab_size, emb_dim]``, then the
            return tensor has shape ``[batch_size, max_time, emb_dim]``.
        """
        if ids is not None:
            if soft_ids is not None:
                raise ValueError(
                    'Must not specify `ids` and `soft_ids` at the same time.')
            ids_rank = ids.dim()
        elif soft_ids is not None:
            ids_rank = soft_ids.dim() - 1
        else:
            raise ValueError('Either `ids` or `soft_ids` must be given.')

        embedding = self._embedding

        if self._hparams.dropout_strategy == 'item_type':
            noise_shape = self._get_noise_shape(self._hparams.dropout_strategy)
            embedding = self._dropout_layer(embedding, noise_shape)

        if ids is not None:
            outputs = F.embedding(ids, embedding, **kwargs)
        else:
            outputs = embedder_utils.soft_embedding_lookup(embedding, soft_ids)

        if self._hparams.dropout_strategy != 'item_type':
            noise_shape = self._get_noise_shape(
                self._hparams.dropout_strategy,
                ids_rank=ids_rank, dropout_input=outputs)
            outputs = self._dropout_layer(outputs, noise_shape)

        return outputs

    @property
    def embedding(self) -> torch.Tensor:
        r"""The embedding tensor, of shape ``[vocab_size] + dim``.
        """
        return self._embedding

    @property
    def dim(self) -> int:
        r"""The embedding dimension.
        """
        return self._dim

    @property
    def vocab_size(self) -> int:
        r"""The vocabulary size.
        """
        return self._vocab_size

    @property
    def num_embeddings(self) -> int:
        r"""The vocabulary size. This interface matches
        :torch_nn:`Embedding`.
        """
        return self._vocab_size

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
