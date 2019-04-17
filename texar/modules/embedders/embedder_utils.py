# Copyright 2018 The Texar Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np

from texar.hyperparams import HParams
from texar.core import layers

__all__ = [
    "default_embedding_hparams",
    "get_embedding",
    "soft_embedding_lookup"
]

def default_embedding_hparams():
    """Returns a `dict` of hyperparameters and default values of a embedder.

     See :meth:`~texar.modules.WordEmbedder.default_hparams` for details.

        .. code-block:: python

            {
                "name": "embedding",
                "dim": 100,
                "initializer": None,
                "regularizer": {
                    "type": "L1L2",
                    "kwargs": {
                        "l1": 0.,
                        "l2": 0.
                    }
                },
                "dropout_rate": 0.,
                "dropout_strategy": 'element',
                "trainable": True,
            }

        Here:

        "name" : str
            Name of the embedding variable.

        "dim" : int or list
            Embedding dimension. Can be a list of integers to yield embeddings
            with dimensionality > 1.

        "initializer" : dict or None
            Hyperparameters of the initializer for the embedding values. An
            example is as

            .. code-block:: python

                {
                    "type": "random_uniform_initializer",
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                }

            which corresponds to :tf_main:`tf.random_uniform_initializer
            <random_uniform_initializer>`, and includes:

            "type" : str or initializer instance
                Name, full path, or instance of the initializer class; Or name
                or full path to a function that returns the initializer class.
                The class or function can be

                - Built-in initializer defined in \
                  :tf_main:`tf.initializers <initializers>`, e.g., \
                  :tf_main:`random_uniform <random_uniform_initializer>` \
                  (a.k.a :class:`tf.random_uniform_initializer`), or \
                  in :mod:`tf`, e.g., :tf_main:`glorot_uniform_initializer \
                  <glorot_uniform_initializer>`, or in \
                  :tf_main:`tf.keras.initializers <keras/initializers>`.
                - User-defined initializer in :mod:`texar.custom`.
                - External initializer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyInitializer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                initializer class or for the function. An initializer is
                created by `initialzier = initializer_class_or_fn(**kwargs)`
                where :attr:`initializer_class_or_fn` is specified in
                :attr:`"type"`.
                Ignored if :attr:`"type"` is an initializer instance.

        "regularizer" : dict
            Hyperparameters of the regularizer for the embedding values. The
            regularizer must be an instance of
            the base :tf_main:`Regularizer <keras/regularizers/Regularizer>`
            class. The hyperparameters include:

            "type" : str or Regularizer instance
                Name, full path, or instance of the regularizer class. The
                class can be

                - Built-in regularizer defined in
                  :tf_main:`tf.keras.regularizers <keras/regularizers>`, e.g.,
                  :tf_main:`L1L2 <keras/regularizers/L1L2>`.
                - User-defined regularizer in :mod:`texar.custom`. The
                  regularizer class should inherit the base class
                  :tf_main:`Regularizer <keras/regularizers/Regularizer>`.
                - External regularizer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyRegularizer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                regularizer class. A regularizer is created by
                calling `regularizer_class(**kwargs)` where
                :attr:`regularizer_class` is specified in :attr:`"type"`.
                Ignored if :attr:`"type"` is a Regularizer instance.

            The default value corresponds to
            :tf_main:`L1L2 <keras/regularizers/L1L2>` with `(l1=0, l2=0)`,
            which disables regularization.

        "dropout_rate" : float
            The dropout rate between 0 and 1. E.g., `dropout_rate=0.1` would
            drop out 10% of the embedding.

        "dropout_strategy" : str
            The dropout strategy. Can be one of the following

            - 'element': The regular strategy that drops individual elements \
              in the embedding vectors.
            - 'item': Drops individual items (e.g., words) entirely. E.g., for \
              the word sequence 'the simpler the better', the strategy can \
              yield '_ simpler the better', where the first `the` is dropped.
            - 'item_type': Drops item types (e.g., word types). E.g., for the \
              above sequence, the strategy can yield '_ simpler _ better', \
              where the word type 'the' is dropped. The dropout will never \
              yield '_ simpler the better' as in the 'item' strategy.

        "trainable" : bool
            Whether the embedding is trainable.
    """
    return {
        "name": "embedding",
        "dim": 100,
        "initializer": None,
        "regularizer": layers.default_regularizer_hparams(),
        "dropout_rate": 0.,
        "dropout_strategy": 'element',
        "trainable": True,
        "@no_typecheck": ["dim"]
    }


def get_embedding(hparams=None,
                  init_value=None,
                  num_embeds=None):
    """Creates embedding variable if not exists.

    Args:
        hparams (dict or HParams, optional): Embedding hyperparameters. Missing
            hyperparameters are set to default values. See
            :func:`~texar.modules.default_embedding_hparams`
            for all hyperparameters and default values.

            If :attr:`init_value` is given, :attr:`hparams["initializer"]`,
            and :attr:`hparams["dim"]` are ignored.
        init_value (Tensor or numpy array, optional): Initial values of the
            embedding variable. If not given, embedding is initialized as
            specified in :attr:`hparams["initializer"]`.
        num_embeds (int, optional): The number of embedding items
            (e.g., vocabulary size). Required if :attr:`init_value` is
            not provided.
        variable_scope (str or VariableScope, optional): Variable scope of
            the embedding variable.

    Returns:
        Variable or Tensor: A 2D `Variable` or `Tensor` of the same shape with
        :attr:`init_value` or of the shape
        :attr:`[num_embeds, hparams["dim"]]`.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_embedding_hparams())
    #regularizer = layers.get_regularizer(hparams["regularizer"])
    if init_value is None:
        initializer = layers.get_initializer(hparams["initializer"])
        dim = hparams["dim"]
        if not isinstance(hparams["dim"], (list, tuple)):
            dim = [dim]
        '''embedding = tf.get_variable(name='w',
                                    shape=[num_embeds] + dim,
                                    initializer=initializer,
                                    regularizer=regularizer,
                                    trainable=hparams["trainable"])'''
        embedding = torch.empty(size=[num_embeds] + dim)
        # initializer should be set by layers.get_initializer
        if initializer:
            embedding = initializer(embedding)
        else:
            embedding = torch.nn.init.xavier_uniform_(embedding)
        #if regularizer:
        #    embedding = regularizer(embedding)
    else:
        '''embedding = tf.get_variable(name='w',
                                    initializer=tf.to_float(init_value),
                                    regularizer=regularizer,
                                    trainable=hparams["trainable"])'''
        embedding = torch.tensor(init_value, dtype=torch.float)
        #if regularizer:
        #    embedding = regularizer(embedding)

    return embedding

def embedding_lookup(embedding, ids, **kwargs):
    outputs = torch.nn.functional.embedding(ids.type(torch.long), embedding, **kwargs)
    return outputs

def soft_embedding_lookup(embedding, soft_ids):
    """Transforms soft ids (e.g., probability distribution over ids) into
    embeddings, by mixing the embedding vectors with the soft weights.

    Args:
        embedding: A Tensor of shape `[num_classes] + embedding-dim` containing
            the embedding vectors. Embedding can have dimensionality > 1, i.e.,
            :attr:`embedding` can be of shape
            `[num_classes, emb_dim_1, emb_dim_2, ...]`
        soft_ids: A Tensor of weights (probabilities) used to mix the
            embedding vectors.

    Returns:
        A Tensor of shape `shape(soft_ids)[:-1] + shape(embedding)[1:]`. For
        example, if `shape(soft_ids) = [batch_size, max_time, vocab_size]`
        and `shape(embedding) = [vocab_size, emb_dim]`, then the return tensor
        has shape `[batch_size, max_time, emb_dim]`.

    Example::

        decoder_outputs, ... = decoder(...)
        soft_seq_emb = soft_embedding_lookup(
            embedding, tf.nn.softmax(decoder_outputs.logits))
    """
    #return tf.tensordot(tf.to_float(soft_ids), embedding, [-1, 0])
    return tensordot_pytorch(torch.tensor(soft_ids, dtype=torch.float), embedding, [-1, 0])

def tensordot_pytorch(a, b, axes=2):
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1
    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")
    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]
    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]
    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)
    res = at.matmul(bt)
    return res.reshape(olda + oldb)
