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
Various helper classes and utilities for RNN decoders.
"""

from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import categorical

from texar.core.cell_wrappers import State
from texar.utils import utils

# from tensorflow.contrib.distributions import RelaxedOneHotCategorical \
#     as GumbelSoftmax
# from texar.modules.embedders.embedder_base import EmbedderBase

# pylint: disable=not-context-manager, too-many-arguments
# pylint: disable=too-many-instance-attributes

__all__ = [
    "default_helper_train_hparams",
    "default_helper_infer_hparams",
    "get_helper",
    "_get_training_helper",
    "TopKSampleEmbeddingHelper",
    "GumbelSoftmaxEmbeddingHelper",
    "SoftmaxEmbeddingHelper"
]

HelperInitTuple = Tuple[torch.ByteTensor, torch.Tensor]
NextInputTuple = Tuple[torch.ByteTensor, torch.Tensor, State]


class Helper:
    """Interface for implementing sampling in seq2seq decoders.

    Helper instances are used by `BasicDecoder`.
    """

    @property
    def sample_ids_shape(self) -> torch.Size:
        r"""Shape of tensor returned by `sample`, excluding the batch dimension.

        Returns a :class:`torch.Size`.
        """
        raise NotImplementedError("sample_ids_shape has not been implemented")

    def initialize(self, inputs: torch.Tensor,
                   sequence_length: torch.LongTensor) -> HelperInitTuple:
        r"""Initialize the current batch.

        Args:
            inputs: A (structure of) input tensors.
            sequence_length: An int32 vector tensor.

        Returns `(initial_finished, initial_inputs)`.
        """
        raise NotImplementedError

    def sample(self, time: int, outputs: torch.Tensor,
               state: State) -> torch.LongTensor:
        r"""Returns `sample_ids`."""
        raise NotImplementedError

    def next_inputs(self, time: int, outputs: torch.Tensor, state: State,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        r"""Returns `(finished, next_inputs, next_state)`."""
        raise NotImplementedError


class TrainingHelper(Helper):
    r"""A helper for use during training.  Only reads inputs.

    Returned sample_ids are the argmax of the RNN output logits.
    """

    def __init__(self, embedding: Optional[nn.Module] = None,
                 time_major: bool = False):
        r"""Initializer.

        Args:
            embedding (optional): The `params` argument of
                :tf_main:`tf.nn.embedding_lookup
                <nn/embedding_lookup>` (e.g., the embedding Tensor); or a
                callable that takes a vector of integer indexes and returns
                respective embedding (e.g., an instance of subclass of
                :class:`~texar.modules.EmbedderBase`).
            time_major: Python bool.  Whether the tensors in `inputs` are time
                major. If `False` (default), they are assumed to be batch major.

        Raises:
            ValueError: if `sequence_length` is not a 1D tensor.
        """
        if embedding is not None:
            if (not callable(embedding) and
                    not isinstance(embedding, torch.Tensor)):
                raise ValueError(
                    "'embedding' must either be a torch.Tensor or a callable.")
            if callable(embedding):
                self._embedding = embedding
            else:
                self._embedding = lambda input: F.embedding(input, embedding)
        else:
            self._embedding = None

        self._time_major = time_major

        # the following are set in `initialize`
        self._zero_inputs = None
        self._inputs = None
        self._sequence_length = None

    @property
    def sample_ids_shape(self) -> torch.Size:
        return torch.Size()  # scalar

    def initialize(self, inputs: torch.Tensor,
                   sequence_length: torch.LongTensor) -> HelperInitTuple:
        if sequence_length.dim() != 1:
            raise ValueError(
                f"Expected 'sequence_length' to be a vector, "
                f"but received shape: {sequence_length.shape}")

        if self._embedding is not None:
            inputs = self._embedding(inputs)
        if not self._time_major:
            inputs = inputs.transpose(0, 1)  # make inputs time major

        self._inputs = inputs
        self._sequence_length = sequence_length
        self._zero_inputs = inputs.new_zeros(inputs[0].size())

        finished: torch.ByteTensor = (sequence_length == 0)
        all_finished = torch.all(finished).item()
        next_inputs = inputs[0] if not all_finished else self._zero_inputs
        return (finished, next_inputs)

    def sample(self, time: int, outputs: torch.Tensor,
               state: State) -> torch.LongTensor:
        sample_ids = torch.argmax(outputs, dim=-1)
        return sample_ids

    def next_inputs(self, time: int, outputs: torch.Tensor, state: State,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        r"""next_inputs_fn for TrainingHelper."""
        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = torch.all(finished).item()

        next_inputs = (self._inputs[next_time] if not all_finished
                       else self._zero_inputs)
        return (finished, next_inputs, state)


class GreedyEmbeddingHelper(Helper):
    """A helper for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding: nn.Module, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor]):
        """Initializer.

        Args:
            embedding: A callable that takes a vector tensor of `ids` (argmax
                ids), or the `params` argument for `embedding_lookup`. The
                returned tensor will be passed to the decoder input.
            start_tokens: 1D LongTensor shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
                not a scalar.
        """
        if not (callable(embedding) or isinstance(embedding, torch.Tensor)):
            raise ValueError(
                "'embedding' must either be a torch.Tensor or a callable.")
        if start_tokens.dim() != 1:
            raise ValueError("start_tokens must be a vector")
        if not isinstance(end_token, int) and end_token.dim() != 0:
            raise ValueError("end_token must be a scalar")

        if callable(embedding):
            self._embedding = embedding
        else:
            self._embedding = lambda input: F.embedding(input, embedding)

        self._start_tokens = start_tokens
        self._batch_size = start_tokens.size(0)
        self._zero_inputs = start_tokens.new_zeros(start_tokens.size())
        if isinstance(end_token, int):
            self._end_token = start_tokens.new_tensor(end_token)
        else:
            self._end_token = end_token

    @property
    def sample_ids_shape(self):
        return torch.Size()

    def initialize(self, inputs: torch.Tensor,
                   sequence_length: torch.LongTensor) -> HelperInitTuple:
        finished = self._start_tokens.new_zeros(
            self._batch_size, dtype=torch.uint8)
        return (finished, self._embedding(self._start_tokens))

    def sample(self, time: int, outputs: torch.Tensor,
               state: State) -> torch.LongTensor:
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not torch.is_tensor(outputs):
            raise TypeError(
                f"Expected outputs to be a single Tensor, got: {type(outputs)}")
        sample_ids = torch.argmax(outputs, dim=-1)
        return sample_ids

    def next_inputs(self, time: int, outputs: torch.Tensor, state: State,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = (sample_ids == self._end_token)
        all_finished = torch.all(finished).item()

        next_inputs = (self._embedding(sample_ids) if not all_finished
                       else self._zero_inputs)
        return (finished, next_inputs, state)


class SampleEmbeddingHelper(GreedyEmbeddingHelper):
    """A helper for use during inference.

    Uses sampling (from a distribution) instead of argmax and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding: nn.Module, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor],
                 softmax_temperature: Optional[float] = None):
        """Initializer.

        Args:
            embedding: A callable that takes a vector tensor of `ids` (argmax
                ids), or the `params` argument for `embedding_lookup`. The
                returned tensor will be passed to the decoder input.
            start_tokens: 1D LongTensor shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
            softmax_temperature: (Optional) `float32` scalar, value to divide
                the logits by before computing the softmax. Larger values (above
                1.0) result in more random samples, while smaller values push
                the sampling distribution towards the argmax. Must be strictly
                greater than 0.  Defaults to 1.0.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
                not a scalar.
        """
        super().__init__(embedding, start_tokens, end_token)
        self._softmax_temperature = softmax_temperature

    def sample(self, time: int, outputs: torch.Tensor,
               state: State) -> torch.LongTensor:
        """sample for SampleEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not torch.is_tensor(outputs):
            raise TypeError(
                f"Expected outputs to be a single Tensor, got: {type(outputs)}")
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        sample_id_sampler = categorical.Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample()

        return sample_ids


def default_helper_train_hparams():
    r"""Returns default hyperparameters of an RNN decoder helper in the training
    phase.

    See also :meth:`~texar.modules.decoders.rnn_decoder_helpers.get_helper`
    for information of the hyperparameters.

    Returns:
        dict: A dictionary with following structure and values:

        .. code-block:: python

            {
                # The `helper_type` argument for `get_helper`, i.e., the name
                # or full path to the helper class.
                "type": "TrainingHelper",

                # The `**kwargs` argument for `get_helper`, i.e., additional
                # keyword arguments for constructing the helper.
                "kwargs": {}
            }
    """
    return {
        "type": "TrainingHelper",
        "kwargs": {}
    }


def default_helper_infer_hparams():
    r"""Returns default hyperparameters of an RNN decoder helper in the
    inference phase.

    See also :meth:`~texar.modules.decoders.rnn_decoder_helpers.get_helper`
    for information of the hyperparameters.

    Returns:
        dict: A dictionary with following structure and values:

        .. code-block:: python

            {
                # The `helper_type` argument for `get_helper`, i.e., the name
                # or full path to the helper class.
                "type": "SampleEmbeddingHelper",

                # The `**kwargs` argument for `get_helper`, i.e., additional
                # keyword arguments for constructing the helper.
                "kwargs": {}
            }
    """
    return {
        "type": "SampleEmbeddingHelper",
        "kwargs": {}
    }


def get_helper(helper_type,
               inputs=None,
               sequence_length=None,
               embedding=None,
               start_tokens=None,
               end_token=None,
               **kwargs):
    r"""Creates a Helper instance.

    Args:
        helper_type: A :tf_main:`Helper <contrib/seq2seq/Helper>` class, its
            name or module path, or a class instance. If a class instance
            is given, it is returned directly.
        inputs (optional): Inputs to the RNN decoder, e.g., ground truth
            tokens for teacher forcing decoding.
        sequence_length (optional): A 1D int Tensor containing the
            sequence length of :attr:`inputs`.
        embedding (optional): A callable that takes a vector tensor of
            indexes (e.g., an instance of subclass of
            :class:`~texar.modules.EmbedderBase`), or the `params` argument
            for `embedding_lookup` (e.g., the embedding Tensor).
        start_tokens (optional): A int Tensor of shape `[batch_size]`,
            the start tokens.
        end_token (optional): A int 0D Tensor, the token that marks end
            of decoding.
        **kwargs: Additional keyword arguments for constructing the helper.

    Returns:
        A helper instance.
    """
    module_paths = [
        'texar.modules.decoders.rnn_decoder_helpers',
        'tensorflow.contrib.seq2seq',
        'texar.custom']
    class_kwargs = {"inputs": inputs,
                    "sequence_length": sequence_length,
                    "embedding": embedding,
                    "start_tokens": start_tokens,
                    "end_token": end_token}
    class_kwargs.update(kwargs)
    return utils.check_or_get_instance_with_redundant_kwargs(
        helper_type, class_kwargs, module_paths)


# pylint: disable-all

def _top_k_logits(logits, k):
    r"""Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return tf.cond(
        tf.equal(k, 0),
        lambda: logits,
        lambda: _top_k(),
    )


class TopKSampleEmbeddingHelper(GreedyEmbeddingHelper):
    r"""A helper for use during inference.

    Samples from `top_k` most likely candidates from a vocab distribution,
    and passes the result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token, top_k=10,
                 softmax_temperature=None, seed=None):
        r"""Initializer.

        Args:
            embedding: A callable that takes a vector tensor of `ids`
                (argmax ids), or the `params` argument for
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
                instance of subclass of :class:`texar.modules.EmbedderBase`.
                The returned tensor will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start
                tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
            top_k: `int32` scalar tensor. Number of top candidates to sample
                from. Must be `>=0`. If set to 0, samples from all candidates
                (i.e., regular random sample decoding).
            softmax_temperature (optional): `float32` scalar, value to
                divide the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples, while smaller values
                push the sampling distribution towards the argmax. Must be
                strictly greater than 0. Defaults to 1.0.
            seed (optional): The sampling seed.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
            not a scalar.
        """
        super(TopKSampleEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token)
        self._top_k = top_k
        self._softmax_temperature = softmax_temperature
        self._seed = seed

    def sample(self, time, outputs, state, name=None):
        r"""sample for SampleEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample from the top_k candidates
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        logits = _top_k_logits(logits, k=self._top_k)

        sample_id_sampler = categorical.Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample(seed=self._seed)

        return sample_ids


class SoftmaxEmbeddingHelper(Helper):
    r"""A helper that feeds softmax probabilities over vocabulary
    to the next step.
    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.

    Args:
        embedding: An embedding argument (:attr:`params`) for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
            instance of subclass of :class:`texar.modules.EmbedderBase`.
            Note that other callables are not acceptable here.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """

    def __init__(self, embedding, start_tokens, end_token, tau,
                 stop_gradient=False, use_finish=True):
        if isinstance(embedding, EmbedderBase):
            embedding = embedding.embedding

        if callable(embedding):
            raise ValueError("`embedding` must be an embedding tensor or an "
                             "instance of subclass of `EmbedderBase`.")
        else:
            self._embedding = embedding
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))

        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._batch_size = tf.size(self._start_tokens)
        self._tau = tau
        self._stop_gradient = stop_gradient
        self._use_finish = use_finish

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._embedding.get_shape()[:1]

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        r"""Returns `sample_id` which is softmax distributions over vocabulary
        with temperature `tau`. Shape = `[batch_size, vocab_size]`
        """
        sample_ids = tf.nn.softmax(outputs / self._tau)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        if self._use_finish:
            hard_ids = tf.argmax(sample_ids, axis=-1, output_type=tf.int32)
            finished = tf.equal(hard_ids, self._end_token)
        else:
            finished = tf.tile([False], [self._batch_size])
        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)
        next_inputs = tf.matmul(sample_ids, self._embedding)
        return (finished, next_inputs, state)


class GumbelSoftmaxEmbeddingHelper(SoftmaxEmbeddingHelper):
    r"""A helper that feeds gumbel softmax sample to the next step.
    Uses the gumbel softmax vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.

    Same as :class:`~texar.modules.SoftmaxEmbeddingHelper` except that here
    gumbel softmax (instead of softmax) is used.

    Args:
        embedding: An embedding argument (:attr:`params`) for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
            instance of subclass of :class:`texar.modules.EmbedderBase`.
            Note that other callables are not acceptable here.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        straight_through (bool): Whether to use straight through gradient
            between time steps. If `True`, a single token with highest
            probability (i.e., greedy sample) is fed to the next step and
            gradient is computed using straight through. If `False` (default),
            the soft gumbel-softmax distribution is fed to the next step.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """

    def __init__(self, embedding, start_tokens, end_token, tau,
                 straight_through=False, stop_gradient=False, use_finish=True):
        super(GumbelSoftmaxEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token, tau, stop_gradient, use_finish)
        self._straight_through = straight_through

    def sample(self, time, outputs, state, name=None):
        r"""Returns `sample_id` of shape `[batch_size, vocab_size]`. If
        `straight_through` is False, this is gumbel softmax distributions over
        vocabulary with temperature `tau`. If `straight_through` is True,
        this is one-hot vectors of the greedy samples.
        """
        sample_ids = tf.nn.softmax(outputs / self._tau)
        sample_ids = GumbelSoftmax(self._tau, logits=outputs).sample()
        if self._straight_through:
            size = tf.shape(sample_ids)[-1]
            sample_ids_hard = tf.cast(
                tf.one_hot(tf.argmax(sample_ids, -1), size), sample_ids.dtype)
            sample_ids = tf.stop_gradient(sample_ids_hard - sample_ids) \
                         + sample_ids
        return sample_ids
