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

# pylint: disable=too-many-arguments, too-many-instance-attributes
# pylint: disable=missing-docstring  # does not support generic classes

from abc import ABC
from typing import Callable, Generic, Optional, Tuple, Type, TypeVar, \
    Union, overload

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from texar.utils import get_args, utils

__all__ = [
    '_convert_embedding',
    'Helper',
    'TrainingHelper',
    'EmbeddingHelper',
    'GreedyEmbeddingHelper',
    'SampleEmbeddingHelper',
    'TopKSampleEmbeddingHelper',
    'SoftmaxEmbeddingHelper',
    'GumbelSoftmaxEmbeddingHelper',
    'default_helper_train_hparams',
    'default_helper_infer_hparams',
    'get_helper',
]
# TODO: Implement `ScheduledEmbeddingTrainingHelper` and
#     `ScheduledOutputTrainingHelper`

HelperInitTuple = Tuple[torch.ByteTensor, torch.Tensor]
NextInputTuple = Tuple[torch.ByteTensor, torch.Tensor]

Embedding = Union[
    torch.Tensor,  # embedding weights
    Callable[[torch.LongTensor], torch.Tensor],  # indices -> embeddings,
    Callable[[List[torch.LongTensor]], torch.Tensor],
]
# indices, position -> embeddings
EmbeddingWithPos = Callable[[torch.LongTensor, torch.LongTensor], torch.Tensor]


@overload
def _convert_embedding(embedding: Embedding) \
        -> Callable[[torch.LongTensor], torch.Tensor]: ...


@overload
def _convert_embedding(embedding: EmbeddingWithPos) -> EmbeddingWithPos: ...


def _convert_embedding(embedding):
    r"""Wrap raw tensors into callables. If the input is already a callable,
    it is returned as is.

    Args:
        embedding (torch.Tensor or callable): the embedding to convert.

    Returns:
        An instance of Embedder or nn.Embedding.
    """
    if callable(embedding):
        return embedding
    elif torch.is_tensor(embedding):
        return lambda x: F.embedding(x, embedding)
    else:
        raise ValueError(
            "'embedding' must either be a torch.Tensor or a callable.")


IDType = TypeVar('IDType', bound=torch.Tensor)


class Helper(Generic[IDType], ABC):
    r"""Interface for implementing sampling in seq2seq decoders.

    Helper instances are used by `BasicDecoder`.
    """

    @property
    def sample_ids_shape(self) -> torch.Size:
        r"""Shape of tensor returned by `sample`, excluding the batch dimension.

        Returns a :class:`torch.Size`.
        """
        raise NotImplementedError("sample_ids_shape has not been implemented")

    def initialize(self, inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor]) \
            -> HelperInitTuple:
        r"""Initialize the current batch.

        Args:
            inputs: A (structure of) input tensors.
            sequence_length: An int32 vector tensor.

        Returns `(initial_finished, initial_inputs)`.
        """
        raise NotImplementedError

    def sample(self, time: int, outputs: torch.Tensor) -> IDType:
        r"""Returns `sample_ids`."""
        raise NotImplementedError

    def next_inputs(self, time: int, outputs: torch.Tensor,
                    sample_ids: IDType) -> NextInputTuple:
        r"""Returns `(finished, next_inputs, next_state)`."""
        raise NotImplementedError


class TrainingHelper(Helper[torch.LongTensor]):
    r"""A helper for use during training.  Only reads inputs.

    Returned sample_ids are the argmax of the RNN output logits.
    """
    _embedding: Optional[Embedding]

    # the following are set in `initialize`
    _inputs: torch.Tensor
    _zero_inputs: torch.Tensor
    _sequence_length: torch.LongTensor

    def __init__(self, embedding: Optional[Embedding] = None,
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
            self._embedding = _convert_embedding(embedding)
        else:
            self._embedding = None

        self._time_major = time_major

    @property
    def sample_ids_shape(self) -> torch.Size:
        return torch.Size()  # scalar

    def initialize(self, inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor]) \
            -> HelperInitTuple:
        if inputs is None:
            raise ValueError("`inputs` cannot be None for TrainingHelper")
        if sequence_length is None:
            raise ValueError(
                "`sequence_length` cannot be None for TrainingHelper")
        inputs: torch.Tensor
        sequence_length: torch.LongTensor

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

    def sample(self, time: int, outputs: torch.Tensor) -> torch.LongTensor:
        del time
        sample_ids = torch.argmax(outputs, dim=-1)
        return sample_ids

    def next_inputs(self, time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        r"""next_inputs_fn for TrainingHelper."""
        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = torch.all(finished).item()

        next_inputs = (self._inputs[next_time] if not all_finished
                       else self._zero_inputs)
        return (finished, next_inputs)


class EmbeddingHelper(Helper[IDType], ABC):
    r"""A generic helper for use during inference.

    Uses output logits for sampling, and passes the result through an embedding
    layer to get the next input.
    """

    _start_inputs: torch.Tensor

    def __init__(self, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor]):
        r"""Initializer.

        Args:
            start_tokens: 1D LongTensor shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
                not a scalar.
        """
        if start_tokens.dim() != 1:
            raise ValueError("start_tokens must be a vector")
        if not isinstance(end_token, int) and end_token.dim() != 0:
            raise ValueError("end_token must be a scalar")

        self._start_tokens = start_tokens
        self._batch_size = start_tokens.size(0)
        if isinstance(end_token, int):
            self._end_token = start_tokens.new_tensor(end_token)
        else:
            self._end_token = end_token

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def initialize(self, inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor]) \
            -> HelperInitTuple:
        finished = self._start_tokens.new_zeros(
            self._batch_size, dtype=torch.uint8)
        return (finished, self._start_inputs)


class SingleEmbeddingHelper(EmbeddingHelper[torch.LongTensor], ABC):
    r"""A generic helper for use during inference.

    Based on :class:`~texar.modules.EmbeddingHelper`, this class returns samples
    that are vocabulary indices, and use the corresponding embeddings as the
    next input. This class also supports callables as embeddings.
    """

    def __init__(self, embedding: Union[Embedding, EmbeddingWithPos],
                 start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor]):
        r"""Initializer

        Args:
            embedding: A callable or the `params` argument for
                `embedding_lookup`.
                If a callable, it can take a vector tensor of `ids` (argmax
                ids), or take two arguments (`ids`, `times`), where `ids` is a
                vector of argmax ids, and `times` is a vector of current time
                steps (i.e., position ids). The latter case can be used when
                attr:`embedding` is a combination of word embedding and position
                embedding.
                The returned tensor will be passed to the decoder input.
            start_tokens: 1D LongTensor shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
                not a scalar.
        """
        super().__init__(start_tokens, end_token)

        self._embedding_fn = _convert_embedding(embedding)

        self._embedding_args_cnt = len(get_args(self._embedding_fn))
        if self._embedding_args_cnt == 1:
            self._start_inputs = self._embedding_fn(  # type: ignore
                self._start_tokens)
        elif self._embedding_args_cnt == 2:
            # Position index is 0 in the beginning
            times = self._start_tokens.new_zeros(self._batch_size,
                    dtype=self._start_tokens.dtype)
            self._start_inputs = self._embedding_fn(  # type: ignore
                self._start_tokens, times)
        else:
            raise ValueError('`embedding` should expect 1 or 2 arguments.')

    @property
    def sample_ids_shape(self) -> torch.Size:
        return torch.Size()

    def next_inputs(self, time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        r"""next_inputs_fn for GreedyEmbeddingHelper."""
        del outputs  # unused by next_inputs_fn
        finished = (sample_ids == self._end_token)
        all_finished = torch.all(finished).item()

        if self._embedding_args_cnt == 1:
            embeddings = self._embedding_fn(sample_ids)  # type: ignore
        else:
            times = self._start_tokens.new_full((self._batch_size,), time + 1)
            embeddings = self._embedding_fn(sample_ids, times)  # type: ignore

        next_inputs = (embeddings if not all_finished else self._start_inputs)
        return (finished, next_inputs)


class GreedyEmbeddingHelper(SingleEmbeddingHelper):
    r"""A helper for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.

    Note that for greedy decoding, Texar's decoders provide a simpler
    interface by specifying `decoding_strategy='infer_greedy'` when calling a
    decoder (see, e.g.,,
    :meth:`RNN decoder <texar.modules.RNNDecoderBase._build>`). In this case,
    use of GreedyEmbeddingHelper is not necessary.
    """

    def __init__(self, embedding: Union[Embedding, EmbeddingWithPos],
                 start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor]):
        r"""Initializer.

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
        super().__init__(embedding, start_tokens, end_token)

    def sample(self, time: int, outputs: torch.Tensor) -> torch.LongTensor:
        r"""sample for GreedyEmbeddingHelper."""
        del time  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not torch.is_tensor(outputs):
            raise TypeError(
                f"Expected outputs to be a single Tensor, got: {type(outputs)}")
        sample_ids = torch.argmax(outputs, dim=-1)
        return sample_ids


class SampleEmbeddingHelper(SingleEmbeddingHelper):
    r"""A helper for use during inference.

    Uses sampling (from a distribution) instead of argmax and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding: Union[Embedding, EmbeddingWithPos],
                 start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor],
                 softmax_temperature: Optional[float] = None):
        r"""Initializer.

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

    def sample(self, time: int, outputs: torch.Tensor) -> torch.LongTensor:
        r"""sample for SampleEmbeddingHelper."""
        del time  # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not torch.is_tensor(outputs):
            raise TypeError(
                f"Expected outputs to be a single Tensor, got: {type(outputs)}")
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        sample_id_sampler = Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample()

        return sample_ids


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    r"""Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0:
        # no truncation
        return logits

    values, _ = torch.topk(logits, k=k)
    min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float('-inf')), logits)


class TopKSampleEmbeddingHelper(SingleEmbeddingHelper):
    r"""A helper for use during inference.

    Samples from `top_k` most likely candidates from a vocab distribution,
    and passes the result through an embedding layer to get the next input.
    """

    def __init__(self, embedding: Embedding, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor], top_k: int = 10,
                 softmax_temperature: Optional[float] = None):
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

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
                not a scalar.
        """
        super().__init__(embedding, start_tokens, end_token)
        # if top_k > self._embedding.num_embeddings:
        #     raise ValueError("'top_k' should not be greater than "
        #                      "the number of embeddings.")
        self._top_k = top_k
        self._softmax_temperature = softmax_temperature

    def sample(self, time: int, outputs: torch.Tensor) -> torch.LongTensor:
        r"""sample for SampleEmbeddingHelper."""
        del time  # unused by sample_fn
        # Outputs are logits, we sample from the top-k candidates
        if not torch.is_tensor(outputs):
            raise TypeError(
                f"Expected outputs to be a single Tensor, got: {type(outputs)}")
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        logits = _top_k_logits(logits, k=self._top_k)

        sample_id_sampler = Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample()

        return sample_ids


class SoftmaxEmbeddingHelper(EmbeddingHelper[torch.Tensor]):
    r"""A helper that feeds softmax probabilities over vocabulary
    to the next step.

    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of :class:`~texar.modules.Helper`. Used as a helper to
    :class:`~texar.modules.RNNDecoderBase` :meth:`_build` in inference mode.
    """

    def __init__(self, embedding: torch.Tensor, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor], tau: float,
                 stop_gradient: bool = False, use_finish: bool = True):
        r"""Initializer

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
        super().__init__(start_tokens, end_token)

        self._embedding = embedding
        self._num_embeddings = embedding.size(0)

        self._start_inputs = F.embedding(self._start_tokens, embedding)

        self._tau = tau
        self._stop_gradient = stop_gradient
        self._use_finish = use_finish

    @property
    def sample_ids_shape(self) -> torch.Size:
        return torch.Size([self._num_embeddings])

    def sample(self, time: int, outputs: torch.Tensor) -> torch.Tensor:
        r"""Returns `sample_id` which is softmax distributions over vocabulary
        with temperature `tau`. Shape = `[batch_size, vocab_size]`
        """
        del time
        sample_ids = torch.softmax(outputs / self._tau, dim=-1)
        return sample_ids

    def next_inputs(self, time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        r"""next_inputs_fn for SoftmaxEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        if self._use_finish:
            hard_ids = torch.argmax(sample_ids, dim=-1)
            finished = (hard_ids == self._end_token)
        else:
            finished = self._start_tokens.new_zeros(
                self._batch_size, dtype=torch.uint8)
        if self._stop_gradient:
            sample_ids = sample_ids.detach()
        next_inputs = torch.matmul(sample_ids, self._embedding)
        return (finished, next_inputs)


class GumbelSoftmaxEmbeddingHelper(SoftmaxEmbeddingHelper):
    r"""A helper that feeds gumbel softmax sample to the next step.

    Uses the gumbel softmax vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of :class:`~texar.modules.Helper`. Used as a helper to
    :class:`~texar.modules.RNNDecoderBase` :meth:`_build` in inference mode.

    Same as :class:`~texar.modules.SoftmaxEmbeddingHelper` except that here
    Gumbel softmax (instead of softmax) is used.
    """

    def __init__(self, embedding: torch.Tensor, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor], tau: float,
                 straight_through: bool = False,
                 stop_gradient: bool = False, use_finish: bool = True):
        r"""Initializer

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
                gradient is computed using straight through. If `False`
                (default), the soft gumbel-softmax distribution is fed to the
                next step.
            stop_gradient (bool): Whether to stop the gradient backpropagation
                when feeding softmax vector to the next step.
            use_finish (bool): Whether to stop decoding once `end_token` is
                generated. If `False`, decoding will continue until
                `max_decoding_length` of the decoder is reached.
        """
        super().__init__(embedding, start_tokens, end_token, tau,
                         stop_gradient, use_finish)
        self._straight_through = straight_through
        # unit-scale, zero-location Gumbel distribution
        self._gumbel = Gumbel(loc=torch.tensor(0.0), scale=torch.tensor(1.0))

    def sample(self, time: int, outputs: torch.Tensor) -> torch.Tensor:
        r"""Returns `sample_id` of shape `[batch_size, vocab_size]`. If
        `straight_through` is False, this is gumbel softmax distributions over
        vocabulary with temperature `tau`. If `straight_through` is True,
        this is one-hot vectors of the greedy samples.
        """
        gumbel_samples = self._gumbel.sample(outputs.size()).to(
            device=outputs.device, dtype=outputs.dtype)
        sample_ids = torch.softmax(
            (outputs + gumbel_samples) / self._tau, dim=-1)
        if self._straight_through:
            argmax_ids = torch.argmax(sample_ids, dim=-1).unsqueeze(1)
            sample_ids_hard = torch.zeros_like(sample_ids).scatter_(
                dim=-1, index=argmax_ids, value=1.0)  # one-hot vectors
            sample_ids = (sample_ids_hard - sample_ids).detach() + sample_ids
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
        'type': 'TrainingHelper',
        'kwargs': {}
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
        'type': 'SampleEmbeddingHelper',
        'kwargs': {}
    }


T = TypeVar('T')  # type argument


def get_helper(helper_type: Union[Type[T], T, str],
               embedding: Optional[Embedding] = None,
               start_tokens: Optional[torch.LongTensor] = None,
               end_token: Optional[Union[int, torch.LongTensor]] = None,
               **kwargs):
    r"""Creates a Helper instance.

    Args:
        helper_type: A :tf_main:`Helper <contrib/seq2seq/Helper>` class, its
            name or module path, or a class instance. If a class instance
            is given, it is returned directly.
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
        'texar.modules.decoders.decoder_helpers',
        'texar.custom']
    class_kwargs = {'embedding': embedding,
                    'start_tokens': start_tokens,
                    'end_token': end_token}
    class_kwargs.update(kwargs)
    return utils.check_or_get_instance_with_redundant_kwargs(
        helper_type, class_kwargs, module_paths)
