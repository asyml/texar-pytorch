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
"""Utils of embedder.
"""
from typing import Optional

import torch

from texar.torch.core import layers
from texar.torch.hyperparams import HParams

__all__ = [
    "default_embedding_hparams",
    "get_embedding",
    "soft_embedding_lookup",
]


def default_embedding_hparams():
    r"""Returns a ``dict`` of hyperparameters and default values of a embedder.

     See :meth:`~texar.torch.modules.WordEmbedder.default_hparams` for details.

        .. code-block:: python

            {
                "name": "embedding",
                "dim": 100,
                "initializer": None,
                "dropout_rate": 0.,
                "dropout_strategy": 'element',
            }

        Here:

        `"name"`: str
            Name of the embedding variable.

        `"dim"`: int or list
            Embedding dimension. Can be a list of integers to yield embeddings
            with dimensionality > 1.

        `"initializer"`: dict or None
            Hyperparameters of the initializer for the embedding values. An
            example is:

            .. code-block:: python

                {
                    "type": torch.nn.init.uniform_,
                    "kwargs": {'a': -0.1, 'b': 0.1}
                }

            which corresponds to :torch_nn:`init.uniform_`, and includes:

            `"type"`: str or function
                Name, full path, or instance of the initializer class or
                initializing function;
                The function can be

                - Built-in initializing function defined in
                  :mod:`torch.nn.init`, e.g., :torch_nn:`init.xavier_uniform`
                  or in :mod:`torch`, e.g., :torch:`rand`.
                - User-defined initializing function in
                  :mod:`texar.torch.custom`.
                - External initializing function or initializer instance.
                  Must provide the full path, e.g.,
                  :attr:`"my_module.MyInitializer"`, or the instance.

            `"kwargs"`: dict
                A dictionary of arguments for constructor of the
                initializer function. An initializer is
                created by ``initializer = initializer_class_or_fn(**kwargs)``
                where :attr:`initializer_class_or_fn` is specified in
                :attr:`"type"`.

        `"dropout_rate"`: float
            The dropout rate between 0 and 1. For example, ``dropout_rate=0.1``
            would zero out 10% of the embedding.

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

    """
    return {
        "name": "embedding",
        "dim": 100,
        "initializer": None,
        "dropout_rate": 0.,
        "dropout_strategy": 'element',
        "@no_typecheck": ["dim"]
    }


def get_embedding(num_embeds: Optional[int] = None,
                  init_value: Optional[torch.Tensor] = None,
                  hparams=None):
    r"""Creates embedding variable if not exists.

    Args:
        hparams (dict or HParams, optional): Embedding hyperparameters. Missing
            hyperparameters are set to default values. See
            :func:`~texar.torch.modules.default_embedding_hparams`
            for all hyperparameters and default values.

            If :attr:`init_value` is given, :attr:`hparams["initializer"]`,
            and :attr:`hparams["dim"]` are ignored.
        init_value (Tensor or numpy array, optional): Initial values of the
            embedding variable. If not given, embedding is initialized as
            specified in :attr:`hparams["initializer"]`.
        num_embeds (int, optional): The number of embedding items
            (e.g., vocabulary size). Required if :attr:`init_value` is
            not provided.

    Returns:
        A 2D :tensor:`Tensor` of the same shape with :attr:`init_value` or of
        the shape ``[num_embeds, hparams["dim"]]``.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_embedding_hparams())
    if init_value is None:
        initializer = layers.get_initializer(
            getattr(hparams, "initializer", None))
        # TODO Shibiao: add regularizer
        dim = hparams["dim"]
        if not isinstance(hparams["dim"], (list, tuple)):
            dim = [dim]
        embedding = torch.empty(size=[num_embeds] + dim)
        # initializer should be set by layers.get_initializer
        if initializer:
            initializer(embedding)
        else:
            torch.nn.init.xavier_uniform_(embedding)
    else:
        if torch.is_tensor(init_value):
            embedding = init_value  # Do not copy the tensor.
        else:
            embedding = torch.tensor(init_value, dtype=torch.float)

    return embedding


def soft_embedding_lookup(embedding, soft_ids):
    r"""Transforms soft ids (e.g., probability distribution over ids) into
    embeddings, by mixing the embedding vectors with the soft weights.

    Args:
        embedding: A Tensor of shape ``[num_classes] + embedding-dim``
            containing the embedding vectors. Embedding can have dimensionality
            > 1, i.e., :attr:`embedding` can be of shape
            ``[num_classes, emb_dim_1, emb_dim_2, ...]``
        soft_ids: A Tensor of weights (probabilities) used to mix the
            embedding vectors.

    Returns:
        A Tensor of shape ``shape(soft_ids)[:-1] + shape(embedding)[1:]``. For
        example, if ``shape(soft_ids) = [batch_size, max_time, vocab_size]``
        and ``shape(embedding) = [vocab_size, emb_dim]``, then the returned
        tensor has shape ``[batch_size, max_time, emb_dim]``.

    Example::

        softmax = torch.nn.Softmax()
        decoder_outputs, ... = decoder(...)
        soft_seq_emb = soft_embedding_lookup(
            embedding, softmax(decoder_outputs.logits))
    """
    return torch.tensordot(soft_ids, embedding, dims=([-1], [0]))
