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
from abc import ABC
from typing import Callable, Generic, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from texar.torch.utils import utils
from texar.torch.utils.dtypes import torch_bool

__all__ = [
    'Helper',
    'TrainingHelper',
    'EmbeddingHelper',
    'GreedyEmbeddingHelper',
    'SampleEmbeddingHelper',
    'TopKSampleEmbeddingHelper',
    'TopPSampleEmbeddingHelper',
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

# indices, position -> embeddings
EmbeddingFn = Callable[[torch.LongTensor, torch.LongTensor], torch.Tensor]

IDType = TypeVar('IDType', bound=torch.Tensor)


# Helper instances are used by :class:`texar.torch.modules.DecoderBase`.
class Helper(Generic[IDType], ABC):
    r"""Interface for implementing sampling in seq2seq decoders.

    Please refer to the documentation for the TensorFlow counterpart
    `tf.contrib.seq2seq.Helper
    <https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/Helper>`_.
    """

    def initialize(self, embedding_fn: EmbeddingFn,
                   inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor]) \
            -> HelperInitTuple:
        r"""Initialize the current batch.

        Args:
            embedding_fn: A function taking input tokens and timestamps,
                returning embedding tensors.
            inputs: Input tensors.
            sequence_length: An int32 vector tensor.

        Returns:
            ``(initial_finished, initial_inputs)``.
        """
        raise NotImplementedError

    def sample(self, time: int, outputs: torch.Tensor) -> IDType:
        r"""Returns ``sample_ids``.
        """
        raise NotImplementedError

    def next_inputs(self, embedding_fn: EmbeddingFn,
                    time: int, outputs: torch.Tensor,
                    sample_ids: IDType) -> NextInputTuple:
        r"""Returns ``(finished, next_inputs, next_state)``.
        """
        raise NotImplementedError


class TrainingHelper(Helper[torch.LongTensor]):
    r"""A helper for use during training. Only reads inputs.

    Returned ``sample_ids`` are the argmax of the RNN output logits.

    Please refer to the documentation for the TensorFlow counterpart
    `tf.contrib.seq2seq.TrainingHelper
    <https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper>`_.

    Args:
        time_major (bool):  Whether the tensors in ``inputs`` are time major.
            If `False` (default), they are assumed to be batch major.
    """
    # the following are set in `initialize`
    _inputs: torch.Tensor
    _zero_inputs: torch.Tensor
    _sequence_length: torch.LongTensor

    def __init__(self, time_major: bool = False):
        self._time_major = time_major

    def initialize(self, embedding_fn: EmbeddingFn,
                   inputs: Optional[torch.Tensor],
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

        if not self._time_major:
            inputs = inputs.transpose(0, 1)  # make inputs time major
        times = torch.arange(
            sequence_length.max(), dtype=torch.long, device=inputs.device)
        times = times.unsqueeze(1).expand(-1, inputs.size(1))
        inputs = embedding_fn(inputs, times)

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

    def next_inputs(self, embedding_fn: EmbeddingFn,
                    time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        del embedding_fn, outputs, sample_ids
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

    Args:
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """

    _start_inputs: torch.Tensor  # set in `initialize`

    def __init__(self, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor]):
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

    def initialize(self, embedding_fn: EmbeddingFn,
                   inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor]) \
            -> HelperInitTuple:
        del inputs, sequence_length
        times = torch.zeros_like(self._start_tokens)
        self._start_inputs = embedding_fn(self._start_tokens, times)
        finished = torch.zeros_like(self._start_tokens, dtype=torch_bool)
        return (finished, self._start_inputs)


class SingleEmbeddingHelper(EmbeddingHelper[torch.LongTensor], ABC):
    r"""A generic helper for use during inference.

    Based on :class:`~texar.torch.modules.EmbeddingHelper`, this class returns
    samples that are vocabulary indices, and use the corresponding embeddings as
    the next input. This class also supports callables as embeddings.

    Args:
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """

    def next_inputs(self, embedding_fn: EmbeddingFn,
                    time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        del outputs  # unused by next_inputs_fn
        finished = (sample_ids == self._end_token)
        all_finished = torch.all(finished).item()

        times = torch.full_like(sample_ids, time + 1)
        embeddings = embedding_fn(sample_ids, times)

        next_inputs = (embeddings if not all_finished else self._start_inputs)
        return (finished, next_inputs)


class GreedyEmbeddingHelper(SingleEmbeddingHelper):
    r"""A helper for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.

    Note that for greedy decoding, Texar's decoders provide a simpler
    interface by specifying ``decoding_strategy='infer_greedy'`` when calling a
    decoder (see, e.g.,,
    :meth:`RNN decoder <texar.torch.modules.RNNDecoderBase.forward>`). In this
    case, use of :class:`GreedyEmbeddingHelper` is not necessary.

    Please refer to the documentation for the TensorFlow counterpart
    `tf.contrib.seq2seq.GreedyEmbeddingHelper
    <https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper>`_.

    Args:
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """

    def sample(self, time: int, outputs: torch.Tensor) -> torch.LongTensor:
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

    Please refer to the documentation for the TensorFlow counterpart
    `tf.contrib.seq2seq.SampleEmbeddingHelper
    <https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/SampleEmbeddingHelper>`_.

    Args:
        embedding: A callable or the ``params`` argument for
            :torch_nn:`functional.embedding`.
            If a callable, it can take a vector tensor of ``ids`` (argmax
            ids), or take two arguments (``ids``, ``times``), where ``ids``
            is a vector of argmax ids, and ``times`` is a vector of current
            time steps (i.e., position ids). The latter case can be used
            when :attr:`embedding` is a combination of word embedding and
            position embedding.
            The returned tensor will be passed to the decoder input.
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.
        softmax_temperature (float, optional): Value to divide the logits by
            before computing the softmax. Larger values (above 1.0) result
            in more random samples, while smaller values push the sampling
            distribution towards the argmax. Must be strictly greater than
            0. Defaults to 1.0.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """

    def __init__(self, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor],
                 softmax_temperature: Optional[float] = None):
        super().__init__(start_tokens, end_token)
        self._softmax_temperature = softmax_temperature

    def sample(self, time: int, outputs: torch.Tensor) -> torch.LongTensor:
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


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    r"""Adapted from
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317#file-top-k-top-p-py-L16-L27"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the
    # threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    for idx in range(logits.size(0)):
        batch_indices = sorted_indices[idx, sorted_indices_to_remove[idx]]
        logits[idx, batch_indices] = float("-inf")
    return logits


class TopKSampleEmbeddingHelper(SingleEmbeddingHelper):
    r"""A helper for use during inference.

    Samples from ``top_k`` most likely candidates from a vocab distribution,
    and passes the result through an embedding layer to get the next input.

    Args:
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.
        top_k (int, optional): Number of top candidates to sample from. Must
            be `>=0`. If set to 0, samples from all candidates (i.e.,
            regular random sample decoding). Defaults to 10.
        softmax_temperature (float, optional): Value to divide the logits by
            before computing the softmax. Larger values (above 1.0) result
            in more random samples, while smaller values push the sampling
            distribution towards the argmax. Must be strictly greater than
            0. Defaults to 1.0.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """

    def __init__(self, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor], top_k: int = 10,
                 softmax_temperature: Optional[float] = None):
        super().__init__(start_tokens, end_token)
        self._top_k = top_k
        self._softmax_temperature = softmax_temperature

    def sample(self, time: int, outputs: torch.Tensor) -> torch.LongTensor:
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


class TopPSampleEmbeddingHelper(SingleEmbeddingHelper):
    r"""A helper for use during inference.

    Samples from candidates that have a cumulative probability of at most `p`
    when arranged in decreasing order, and passes the result through an
    embedding layer to get the next input. This is also named as
    "*Nucleus Sampling*" as proposed in the paper
    "*The Curious Case of Neural Text Degeneration(Holtzman et al.)*".

    Args:
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.
        p (float, optional): A value used to filter out tokens whose cumulative
            probability is greater than `p` when arranged in decreasing order of
            probabilities. Must be between [0, 1.0]. If set to 1, samples from
            all candidates (i.e., regular random sample decoding). Defaults to
            0.5.
        softmax_temperature (float, optional): Value to divide the logits by
            before computing the softmax. Larger values (above 1.0) result
            in more random samples, while smaller values push the sampling
            distribution towards the argmax. Must be strictly greater than
            0. Defaults to 1.0.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """

    def __init__(self, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor], p: float = 0.9,
                 softmax_temperature: Optional[float] = None):
        super().__init__(start_tokens, end_token)
        self._p = p
        self._softmax_temperature = softmax_temperature

    def sample(self, time: int, outputs: torch.Tensor) -> torch.LongTensor:
        del time  # unused by sample_fn
        # Outputs are logits, we sample from tokens with cumulative
        # probability at most p when arranged in decreasing order
        if not torch.is_tensor(outputs):
            raise TypeError(
                f"Expected outputs to be a single Tensor, got: {type(outputs)}")
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        logits = _top_p_logits(logits, p=self._p)

        sample_id_sampler = Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample()

        return sample_ids


class SoftmaxEmbeddingHelper(EmbeddingHelper[torch.Tensor]):
    r"""A helper that feeds softmax probabilities over vocabulary
    to the next step.

    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of :class:`~texar.torch.modules.Helper`. Used as a helper to
    :class:`~texar.torch.modules.RNNDecoderBase` in inference mode.

    Args:
        embedding: A callable or the ``params`` argument for
            :torch_nn:`functional.embedding`.
            If a callable, it can take a vector tensor of ``ids`` (argmax
            ids), or take two arguments (``ids``, ``times``), where ``ids``
            is a vector of argmax ids, and ``times`` is a vector of current
            time steps (i.e., position ids). The latter case can be used
            when :attr:`embedding` is a combination of word embedding and
            position embedding.
            The returned tensor will be passed to the decoder input.
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.
        tau: A float scalar tensor, the softmax temperature.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once :attr:`end_token`
            is generated. If `False`, decoding will continue until
            :attr:`max_decoding_length` of the decoder is reached.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """

    def __init__(self, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor], tau: float,
                 stop_gradient: bool = False, use_finish: bool = True):
        super().__init__(start_tokens, end_token)

        self._tau = tau
        self._stop_gradient = stop_gradient
        self._use_finish = use_finish

    def sample(self, time: int, outputs: torch.Tensor) -> torch.Tensor:
        r"""Returns ``sample_id`` which is softmax distributions over vocabulary
        with temperature :attr:`tau`. Shape = ``[batch_size, vocab_size]``.
        """
        del time
        sample_ids = torch.softmax(outputs / self._tau, dim=-1)
        return sample_ids

    def next_inputs(self, embedding_fn: EmbeddingFn,
                    time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor) -> NextInputTuple:
        del outputs  # unused by next_inputs_fn
        if self._use_finish:
            hard_ids = torch.argmax(sample_ids, dim=-1)
            finished = (hard_ids == self._end_token)
        else:
            finished = torch.zeros_like(self._start_tokens, dtype=torch_bool)
        if self._stop_gradient:
            sample_ids = sample_ids.detach()

        indices = torch.arange(sample_ids.size(-1), device=sample_ids.device)
        times = torch.full_like(indices, time + 1)
        embeddings = embedding_fn(indices, times)

        next_inputs = torch.matmul(sample_ids, embeddings)
        return (finished, next_inputs)


class GumbelSoftmaxEmbeddingHelper(SoftmaxEmbeddingHelper):
    r"""A helper that feeds Gumbel softmax sample to the next step.

    Uses the Gumbel softmax vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of :class:`~texar.torch.modules.Helper`. Used as a helper to
    :class:`~texar.torch.modules.RNNDecoderBase` in inference mode.

    Same as :class:`~texar.torch.modules.SoftmaxEmbeddingHelper` except that
    here Gumbel softmax (instead of softmax) is used.

    Args:
        embedding: A callable or the ``params`` argument for
            :torch_nn:`functional.embedding`.
            If a callable, it can take a vector tensor of ``ids`` (argmax
            ids), or take two arguments (``ids``, ``times``), where ``ids``
            is a vector of argmax ids, and ``times`` is a vector of current
            time steps (i.e., position ids). The latter case can be used
            when :attr:`embedding` is a combination of word embedding and
            position embedding.
            The returned tensor will be passed to the decoder input.
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.
        tau: A float scalar tensor, the softmax temperature.
        straight_through (bool): Whether to use straight through gradient
            between time steps. If `True`, a single token with highest
            probability (i.e., greedy sample) is fed to the next step and
            gradient is computed using straight through. If `False`
            (default), the soft Gumbel-softmax distribution is fed to the
            next step.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once :attr:`end_token`
            is generated. If `False`, decoding will continue until
            :attr:`max_decoding_length` of the decoder is reached.

    Raises:
        ValueError: if :attr:`start_tokens` is not a 1D tensor or
            :attr:`end_token` is not a scalar.
    """

    def __init__(self, start_tokens: torch.LongTensor,
                 end_token: Union[int, torch.LongTensor], tau: float,
                 straight_through: bool = False,
                 stop_gradient: bool = False, use_finish: bool = True):
        super().__init__(start_tokens, end_token, tau,
                         stop_gradient, use_finish)
        self._straight_through = straight_through
        # unit-scale, zero-location Gumbel distribution
        self._gumbel = Gumbel(loc=torch.tensor(0.0), scale=torch.tensor(1.0))

    def sample(self, time: int, outputs: torch.Tensor) -> torch.Tensor:
        r"""Returns ``sample_id`` of shape ``[batch_size, vocab_size]``. If
        :attr:`straight_through` is `False`, this contains the Gumbel softmax
        distributions over vocabulary with temperature :attr:`tau`. If
        :attr:`straight_through` is `True`, this contains one-hot vectors of
        the greedy samples.
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

    See also
    :meth:`~texar.torch.modules.decoders.rnn_decoder_helpers.get_helper` for
    information of the hyperparameters.

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

    See also
    :meth:`~texar.torch.modules.decoders.rnn_decoder_helpers.get_helper` for
    information of the hyperparameters.

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
               start_tokens: Optional[torch.LongTensor] = None,
               end_token: Optional[Union[int, torch.LongTensor]] = None,
               **kwargs):
    r"""Creates a Helper instance.

    Args:
        helper_type: A :class:`~texar.torch.modules.Helper` class, its
            name or module path, or a class instance. If a class instance
            is given, it is returned directly.
        start_tokens: 1D :tensor:`LongTensor` shaped ``[batch_size]``,
            representing the start tokens for each sequence in batch.
        end_token: Python int or scalar :tensor:`LongTensor`, denoting the
            token that marks end of decoding.
        **kwargs: Additional keyword arguments for constructing the helper.

    Returns:
        A helper instance.
    """
    module_paths = [
        'texar.torch.modules.decoders.decoder_helpers',
        'texar.torch.custom']
    class_kwargs = {'start_tokens': start_tokens,
                    'end_token': end_token}
    class_kwargs.update(kwargs)
    return utils.check_or_get_instance_with_redundant_kwargs(
        helper_type, class_kwargs, module_paths)
