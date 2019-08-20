# Adapted from the Tensor2Tensor's implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2019 Texar
# ==============================================================================
"""
Implementation of beam search with penalties.

Adapted from:
    `https://github.com/tensorflow/tensor2tensor/blob/eb048f69c7ea860324122b87cb9caf59c52a27f3/tensor2tensor/utils/beam_search.py`
"""
from typing import Any, Callable, Optional, Tuple, TypeVar, overload

import torch

from texar.torch.utils import map_structure, torch_bool

__all__ = [
    'beam_search',
]

State = TypeVar('State')

# Default value for INF
INF = 1.0 * 1e7


def gather_nd(params: Any, indices: torch.Tensor) -> Any:
    if not isinstance(params, torch.Tensor):
        return params
    assert len(indices.size()) == 3
    orig_size = params.size()
    index = indices[:, :, 1].view(-1) + indices[:, :, 0].view(-1) * orig_size[1]
    ret = torch.index_select(
        params.view(-1, *params.size()[2:]), dim=0, index=index
    )
    ret = ret.view(orig_size[0], indices.size(1), *orig_size[2:])

    return ret


def _merge_beam_dim(tensor: Any) -> Any:
    r"""Reshapes first two dimensions in to single dimension.

    Args:
        tensor: Tensor to reshape of shape `[A, B, ...]`.

    Returns:
        Reshaped tensor of shape `[A * B, ...]`.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    shape = list(tensor.size())
    shape[0] *= shape[1]  # batch -> batch * beam_size
    shape.pop(1)  # Remove beam dim
    return tensor.view(tuple(shape))


def _unmerge_beam_dim(tensor: Any, batch_size: int,
                      beam_size: int) -> Any:
    r"""Reshapes first dimension back to `[batch_size, beam_size]`.

    Args:
        tensor: Tensor to reshape of shape `[batch_size * beam_size, ...]`.
        batch_size: int, original batch size.
        beam_size: int, original beam size.

    Returns:
        Reshaped tensor of shape `[batch_size, beam_size, ...]`.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    shape = list(tensor.size())
    new_shape = [batch_size] + [beam_size] + shape[1:]
    return tensor.view(tuple(new_shape))


def _expand_to_beam_size(tensor: Any,
                         beam_size: int) -> Any:
    r"""Tiles a given tensor by :attr:`beam_size`.

    Args:
        tensor: tensor to tile. Shape: `[batch_size, ...]`.
        beam_size: How much to tile the tensor by.

    Returns:
        Tiled tensor of shape `[batch_size, beam_size, ...]`.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    tensor = torch.unsqueeze(tensor, dim=1)
    tile_dims = [1] * len(tensor.size())
    tile_dims[1] = beam_size

    return tensor.repeat(tuple(tile_dims))


def log_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def compute_batch_indices(batch_size: int, beam_size: int) -> torch.LongTensor:
    r"""Computes the i-th coordinate that contains the batch index for
    gathers.

    The batch index tensor is a tensor like `[[0,0,0,0,],[1,1,1,1],..]`.
    It says which batch the beam item is in. This will create the first
    dimension of the 2D coordinates needed for the gather.

    Args:
        batch_size: Batch size
        beam_size: Size of the beam.

    Returns:
        `[batch_size, beam_size]` tensor of ids.
    """
    batch_pos = torch.arange(batch_size)
    batch_pos = batch_pos.view(-1, 1).expand(batch_size, beam_size)
    return batch_pos


def compute_topk_scores_and_seq(
    sequences: torch.LongTensor,
    scores: torch.Tensor,
    scores_to_gather: torch.Tensor,
    flags: torch.ByteTensor,
    beam_size: int,
    batch_size: int,
    states_to_gather: Optional[State] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.ByteTensor, Optional[State]]:
    r"""Given sequences and scores, will gather the top-k (`k = beam`) size
    sequences.

    This function is used to grow alive, and finished. It takes sequences,
    scores, and flags, and returns the top k from sequence
    :attr:`scores_to_gather`, and flags based on the values in scores.

    Args:
        sequences: Tensor of sequences that we need to gather from.
            Shape: `[batch_size, beam_size, seq_length]`.
        scores: Tensor of scores for each sequence in sequences. We will use
            these to compute the top-k. Shape: `[batch_size, beam_size]`.
        scores_to_gather: Tensor of scores for each sequence in sequences.
            Shape: `[batch_size, beam_size]`.
            We will return the gathered scores from here.
            Scores to gather is different from scores because for
            grow_alive, we will need to return log-probabilities, while for
            grow_finished, we will need to return the length penalized
            scores.
        flags: Tensor of booleans for sequences that say whether a sequence
            has reached `EOS`.
        beam_size: int
        batch_size: int
        states_to_gather: (possibly nested structure of) decoding states.

    :returns: Tuple of:

        - `topk_seq`: `[batch_size, beam_size, decode_length]`.
        - `topk_gathered_scores`: `[batch_size, beam_size]`.
        - `topk_finished_flags`: `[batch_size, beam_size]`.
    """
    # by default top-k is for the last dimension
    _, topk_indexes = torch.topk(scores, k=beam_size)
    # The next three steps are to create coordinates for torch.gather_nd to
    # pull out the top-k sequences from sequences based on scores.
    # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    # batch the beam item is in. This will create the i of the i,j
    # coordinate needed for the gather
    batch_pos = compute_batch_indices(batch_size, beam_size)
    batch_pos = batch_pos.to(device=topk_indexes.device)
    # top coordinates will give us the actual coordinates to do the gather.
    # stacking will create a tensor of dimension batch * beam * 2, where
    # the last dimension contains the i,j gathering coordinates.
    top_coordinates = torch.stack([batch_pos, topk_indexes], dim=2)

    # Gather up the highest scoring sequences.
    topk_seq = gather_nd(sequences, top_coordinates)
    topk_flags = gather_nd(flags, top_coordinates)
    topk_gathered_scores = gather_nd(scores_to_gather, top_coordinates)
    if states_to_gather is not None:
        topk_gathered_states = map_structure(
            lambda state: gather_nd(state, top_coordinates), states_to_gather
        )
    else:
        topk_gathered_states = states_to_gather
    return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


# TODO: Remove these once pylint supports function stubs.
# pylint: disable=unused-argument,function-redefined

@overload
def beam_search(
    symbols_to_logits_fn: Callable[[torch.Tensor, State],
                                   Tuple[torch.Tensor, State]],
    initial_ids: torch.LongTensor,
    beam_size: int,
    decode_length: int,
    vocab_size: int,
    alpha: float,
    eos_id: int,
    states: State,
    stop_early: bool = True) -> Tuple[torch.LongTensor, torch.Tensor]: ...


@overload
def beam_search(
    symbols_to_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    initial_ids: torch.LongTensor,
    beam_size: int,
    decode_length: int,
    vocab_size: int,
    alpha: float,
    eos_id: int,
    states: Optional[State] = None,
    stop_early: bool = True) -> Tuple[torch.LongTensor, torch.Tensor]: ...

# pylint: enable=unused-argument


def beam_search(
    symbols_to_logits_fn,
    initial_ids,
    beam_size,
    decode_length,
    vocab_size,
    alpha,
    eos_id,
    states=None,
    stop_early=True,
):
    r"""Beam search with length penalties.

    Requires a function that can take the currently decoded symbols and
    return the logits for the next symbol. The implementation is inspired
    by https://arxiv.org/abs/1609.08144.

    Variables used within this function follow the naming pattern:
    `(alive|finished)_topk_(seq,scores)`.

    Variables marked `alive` represent the new beam sequences that will be
    processed in the next step.    Variables marked `finished` represent
    the completed beam sequences, which may be padded with 0 if no beams
    finished.

    Variables marked `seq` store the full beam sequence for the time step.
    Variables marked `scores` store the sequence's final log scores.

    The beam search steps will be processed sequentially in order, so when
    capturing observed from these operations, tensors, clients can make
    assumptions about which step is being recorded.

    Args:
        symbols_to_logits_fn: Interface to the model, to provide logits.
            Should take `[batch_size, decoded_ids]` and return
            `[batch_size, vocab_size]`.
        initial_ids: LongTensor of shape `[batch_size]`. IDs to start off the
            decoding, this will be the first thing handed to
            :attr:`symbols_to_logits_fn` (after expanding to beam size).
        beam_size: Size of the beam.
        decode_length: Number of steps to decode for.
        vocab_size: Size of the vocab, must equal the size of the logits
            returned by :attr:`symbols_to_logits_fn`.
        alpha: alpha for length penalty.
        eos_id: ID for end of sentence.
        states: (possibly nested structure of) decoding states.
        stop_early: a boolean - stop once best sequence is provably
            determined.

    Returns:
        Tuple of

        - decoded beams (shape: `[batch_size, beam_size, decode_length]`)
        - decoding probabilities (shape: `[batch_size, beam_size]`)
    """

    batch_size = initial_ids.size()[0]

    # Assume initial_ids are prob 1.0
    initial_log_probs = torch.Tensor(
        [[0.0] + [-float("inf")] * (beam_size - 1)]
    )  # [1, beam_size]
    initial_log_probs = initial_log_probs.to(device=initial_ids.device)
    # Expand to beam_size (batch_size, beam_size)
    alive_log_probs = initial_log_probs.repeat((batch_size, 1))

    # Expand each batch and state to beam_size
    alive_seq = _expand_to_beam_size(initial_ids, beam_size)
    alive_seq = torch.unsqueeze(alive_seq, dim=2)
    # (batch_size, beam_size, 1)

    if states is not None:
        states = map_structure(
            lambda state: _expand_to_beam_size(state, beam_size), states
        )

    # Finished will keep track of all the sequences that have finished so
    # far
    # Finished log-probs will be negative infinity in the beginning
    # finished_flags will keep track of booleans
    finished_seq = torch.zeros(alive_seq.size(), dtype=torch.long)
    # Setting the scores of the initial to negative infinity.
    finished_scores = torch.full((batch_size, beam_size), -INF)
    finished_flags = torch.zeros((batch_size, beam_size), dtype=torch_bool)

    finished_seq = finished_seq.to(device=initial_ids.device)
    finished_scores = finished_scores.to(device=initial_ids.device)
    finished_flags = finished_flags.to(device=initial_ids.device)

    def grow_finished(
        finished_seq: torch.LongTensor,
        finished_scores: torch.Tensor,
        finished_flags: torch.ByteTensor,
        curr_seq: torch.LongTensor,
        curr_scores: torch.Tensor,
        curr_finished: torch.ByteTensor,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.ByteTensor]:
        r"""Given sequences and scores, will gather the top-k (`k = beam`) size
        sequences.

        Args:
            finished_seq: Finished sequences.
                Shape: `[batch_size, beam_size, current_decoded_length]`.
            finished_scores: Scores for each finished sequences.
                Shape: `[batch_size, beam_size]`.
            finished_flags: Finished flags for each of these sequences.
                Shape: `[batch_size, beam_size]`
            curr_seq: Top-k sequences that has been grown by one
                position.
                Shape: `[batch_size, beam_size, current_decoded_length]`.
            curr_scores: Scores for each of the top-k sequences.
                Shape: `[batch_size, beam_size]`.
            curr_finished: Finished flags for each of the top-k sequences.
                Shape: `[batch_size, beam_size]`.

        Returns:
            Tuple of

            - Top-k sequences based on scores.
            - Log-probabilities of these sequences.
            - Finished flags of these sequences.
        """
        # First append a column of 0'ids to finished to make the same
        # length with finished scores
        _appended = torch.zeros(batch_size, beam_size, 1, dtype=torch.long)
        _appended = _appended.to(device=finished_seq.device)
        finished_seq = torch.cat([finished_seq, _appended], dim=2)

        # Set the scores of the unfinished seq in curr_seq to large
        # negative values
        curr_scores = curr_scores + (1.0 - curr_finished.float()) * -INF
        # concatenating the sequences and scores along beam axis
        curr_finished_seq = torch.cat([finished_seq, curr_seq], dim=1)
        curr_finished_scores = torch.cat([finished_scores, curr_scores], dim=1)
        curr_finished_flags = torch.cat([finished_flags, curr_finished], dim=1)
        next_seq, next_scores, next_flags, _ = compute_topk_scores_and_seq(
            curr_finished_seq,
            curr_finished_scores,
            curr_finished_scores,
            curr_finished_flags,
            beam_size,
            batch_size,
        )
        return next_seq, next_scores, next_flags

    def grow_alive(
        curr_seq: torch.LongTensor, curr_scores: torch.Tensor,
            curr_log_probs: torch.Tensor, curr_finished: torch.ByteTensor,
            states: Optional[State]
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.ByteTensor,
               Optional[State]]:
        r"""Given sequences and scores, will gather the top k=beam size
        sequences.

        Args:
            curr_seq: Current top-k sequences that has been grown by one
                position.
                Shape: `[batch_size, beam_size, i + 1]`.
            curr_scores: Scores for each of these sequences.
                Shape: `[batch_size, beam_size]`.
            curr_log_probs: Log-probabilities for each of these sequences.
                Shape: `[batch_size, beam_size]`.
            curr_finished: Finished flags for each of these sequences.
                Shape: `[batch_size, beam_size]`.
            states: (possibly nested structure of) decoding states.

        :returns: Tuple of:

            - Top-k sequences based on scores.
            - Log-probabilities of these sequences.
            - Finished flags of these sequences.
            - Decoding states for these sequences.
        """
        # Set the scores of the finished seq in curr_seq to large negative
        # values
        curr_scores = curr_scores + curr_finished.float() * -INF
        return compute_topk_scores_and_seq(
            curr_seq,
            curr_scores,
            curr_log_probs,
            curr_finished,
            beam_size,
            batch_size,
            states,
        )

    def grow_topk(
        i: int, alive_seq: torch.LongTensor, alive_log_probs: torch.Tensor,
        states: Optional[State]
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor,
               torch.ByteTensor, Optional[State]]:
        r"""Inner beam search loop.

        This function takes the current alive sequences, and grows them to
        top-k sequences where `k = 2 * beam`. We use `2 * beam` because we could
        have `beam_size` number of sequences that might hit `<EOS>` and there
        will be no alive sequences to continue. With `2 * beam_size`, this
        will not happen. This relies on the assumption the vocab size is >
        beam size. If this is true, we'll have at least `beam_size` non-`<EOS>`
        extensions if we extract the next top `2 * beam` words.
        Length penalty is given by :math:`(5+len(decode)/6) ^ -\alpha`.

        Please refer to https://arxiv.org/abs/1609.08144.

        Args:
            i: loop index
            alive_seq: Top-k sequences decoded so far.
                Shape: `[batch_size, beam_size, i + 1]`.
            alive_log_probs: Log-probabilities of these sequences.
                Shape: `[batch_size, beam_size]`
            states: (possibly nested structure of) decoding states.

        :returns: Tuple of:

            - Top-k sequences extended by the next word.
            - Log-probabilities of these sequences,
            - The scores with length penalty of these sequences,
            - Flags indicating which of these sequences have finished
              decoding.
            - Transformed decoding states with same structure as :attr:`state`.
        """
        # Get the logits for all the possible next symbols
        flat_ids = alive_seq.view(batch_size * beam_size, -1)

        # (batch_size * beam_size, decoded_length)
        if states is not None:
            flat_states = map_structure(_merge_beam_dim, states)
            flat_logits, flat_states = symbols_to_logits_fn(
                flat_ids, flat_states
            )
            states = map_structure(
                lambda t: _unmerge_beam_dim(t, batch_size, beam_size),
                flat_states,
            )
        else:
            flat_logits = symbols_to_logits_fn(flat_ids)
        logits = flat_logits.view(batch_size, beam_size, -1)

        # Convert logits to normalized log-probs
        candidate_log_probs = log_prob_from_logits(logits)

        # Multiply the probabilities by the current probabilities of the
        # beam.
        # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
        log_probs = candidate_log_probs + alive_log_probs.unsqueeze(dim=2)

        length_penalty = ((5.0 + float(i + 1)) / 6.0) ** alpha

        curr_scores = log_probs / length_penalty
        # Flatten out (beam_size, vocab_size) probs in to a list of
        # possibilities
        flat_curr_scores = curr_scores.view(-1, beam_size * vocab_size)

        topk_scores, topk_ids = torch.topk(flat_curr_scores, k=beam_size * 2)
        # Recovering the log-probs because we will need to send them back
        topk_log_probs = topk_scores * length_penalty

        # Work out what beam the top probabilities are in.
        topk_beam_index = topk_ids / vocab_size
        topk_ids %= vocab_size  # Un-flatten the ids

        # The next three steps are to create coordinates for torch.gather_nd
        # to pull out the correct sequences from id's that we need to grow.
        # We will also use the coordinates to gather the booleans of the
        # beam items that survived.
        batch_pos = compute_batch_indices(batch_size, beam_size * 2)
        batch_pos = batch_pos.to(device=topk_beam_index.device)
        # top beams will give us the actual coordinates to do the gather.
        # stacking will create a tensor of dimension batch * beam * 2,
        # where the last dimension contains the i,j gathering coordinates.
        topk_coordinates = torch.stack([batch_pos, topk_beam_index], dim=2)
        # [batch_size, beam_size, 2]

        topk_seq = gather_nd(alive_seq, topk_coordinates)

        if states is not None:
            states = map_structure(
                lambda state: gather_nd(state, topk_coordinates), states
            )

        # Append the most probable alive
        topk_seq = torch.cat([topk_seq, topk_ids.unsqueeze(dim=2)], dim=2)

        topk_finished = topk_ids == eos_id

        return topk_seq, topk_log_probs, topk_scores, topk_finished, states

    def inner_loop(
        i: int,
        alive_seq: torch.LongTensor,
        alive_log_probs: torch.Tensor,
        finished_seq: torch.LongTensor,
        finished_scores: torch.Tensor,
        finished_flags: torch.ByteTensor,
        states: Optional[State],
    ) -> Tuple[int, torch.LongTensor, torch.Tensor, torch.LongTensor,
               torch.Tensor, torch.ByteTensor, Optional[State]]:
        r"""Inner beam search loop.

        There are three groups of tensors: `alive`, `finished`, and `top-k`.

        - The `alive` group contains information about the current alive
          sequences.
        - The `top-k` group contains information about `alive + top_k`
          current decoded words.
        - The `finished` group contains information about finished sentences,
          that is, the ones that have decoded to `<EOS>`. These are what we
          return.

        The general beam search algorithm is as follows:

            While not terminated (please refer to termination condition):

            1. Grow the current `alive` to get `beam * 2` top-k sequences.
            2. Among the `top-k`, move the top `beam_size` ones that haven't
               reached `EOS` into `alive`.
            3. Among the `top-k`, move the top `beam_size` ones have reached
               `EOS` into `finished`.

            Repeat

        To make things simple with using fixed size tensors, we will end
        up inserting unfinished sequences into finished in the beginning.
        To prevent that we add `-INF` to the score of the unfinished
        sequence so that when a true finished sequence does appear, it
        will have a higher score than all the unfinished ones.

        Args:
            i: Loop index
            alive_seq: Topk sequences decoded so far
                Shape: `[batch_size, beam_size, i + 1]`.
            alive_log_probs: Log-probabilities of the beams.
                Shape: `[batch_size, beam_size]`
            finished_seq: Current finished sequences.
                Shape: `[batch_size, beam_size, i+1]`.
            finished_scores: Scores for each of these sequences.
                Shape: `[batch_size, beam_size]`.
            finished_flags: Finished flags for each of these sequences.
                Shape: `[batch_size, beam_size]`
            states: (possibly nested structure of) decoding states.

        :returns: Tuple of:

            - Incremented loop index.
            - New `alive` sequences.
            - Log-probabilities of the `alive` sequences.
            - New `finished` sequences.
            - Scores of the `finished` sequences.
            - Flags indicating which sequences in `finished` has reached `EOS`.
            - Final decoding states with same structure as :attr:`state`.
        """

        # Each inner loop, we carry out three steps:
        # 1. Get the current top-k items.
        # 2. Extract the ones that have finished and haven't finished
        # 3. Recompute the contents of finished based on scores.
        topk_seq, topk_log_probs, topk_scores, topk_finished, \
                states = grow_topk(i, alive_seq, alive_log_probs, states)

        alive_seq, alive_log_probs, _, states = grow_alive(
            topk_seq, topk_scores, topk_log_probs, topk_finished, states
        )
        finished_seq, finished_scores, finished_flags = grow_finished(
            finished_seq,
            finished_scores,
            finished_flags,
            topk_seq,
            topk_scores,
            topk_finished,
        )

        return (
            i + 1,
            alive_seq,
            alive_log_probs,
            finished_seq,
            finished_scores,
            finished_flags,
            states,
        )

    def _is_finished(
        i: int,
        alive_log_probs: torch.Tensor,
        finished_scores: torch.Tensor
    ) -> bool:
        r"""Check termination condition.

        We terminate when we decoded up to `decode_length` or the lowest
        scoring item in finished has a greater score that the highest probable
        item in alive divided by the max length penalty.

        Args:
            i: Loop index
            alive_log_probs: Log-probabilities of the beams.
                Shape: `[batch_size, beam_size]`.
            finished_scores: Scores for each of these sequences.
                Shape: `[batch_size, beam_size]`.

        Returns:
            Bool.
        """
        max_length_penalty = ((5.0 + float(decode_length)) / 6.0) ** alpha
        # The best possible score of the most likely alive sequence
        lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

        if not stop_early:
            # by considering the min score (in the top N beams) we ensure that
            # the decoder will keep decoding until there is at least one beam
            # (in the top N) that can be improved (w.r.t. the alive beams).
            # any unfinished beam will have score -INF - thus the min
            # will always be -INF if there is at least one unfinished beam -
            # which means the bound_is_met condition cannot be true in this
            # case.
            lowest_score_of_finished_in_finished = torch.min(finished_scores)
        else:
            # by taking the max score we only care about the first beam;
            # as soon as this first beam cannot be beaten from the alive beams
            # the beam decoder can stop.
            # similarly to the above, if the top beam is not completed, its
            # finished_score is -INF, thus it will not activate the
            # bound_is_met condition. (i.e., decoder will keep going on).
            # note we need to find the max for every sequence eparately - so,
            # we need to keep the batch dimension (see axis=1)
            lowest_score_of_finished_in_finished, _ = torch.max(finished_scores,
                                                                dim=1)

        bound_is_met = (
            (lowest_score_of_finished_in_finished > lower_bound_alive_scores)
            .all()
            .item()
        )

        ret = (i < decode_length) & (~bound_is_met)

        return ret

    step = 0
    while _is_finished(step, alive_log_probs, finished_scores):
        step, alive_seq, alive_log_probs, finished_seq, finished_scores, \
                finished_flags, states = inner_loop(
            step,
            alive_seq,
            alive_log_probs,
            finished_seq,
            finished_scores,
            finished_flags,
            states,
        )

    # Accounting for corner case: It's possible that no sequence in alive
    # for a particular batch item ever reached EOS. In that case, we
    # should just copy the contents of alive for that batch item. tf
    # reduce_any(finished_flags, 1)
    # if 0, means that no sequence for that batch index had reached EOS.
    # We need to do the same for the scores as well.

    ret_seq, ret_scores = [], []
    for idx, flag_per_instance in enumerate(finished_flags.any(dim=1).tolist()):
        if flag_per_instance:
            ret_seq.append(finished_seq[idx])
            ret_scores.append(finished_scores[idx])
        else:
            ret_seq.append(alive_seq[idx])
            ret_scores.append(alive_log_probs[idx])

    ret_seq = torch.stack(ret_seq, dim=0)
    ret_scores = torch.stack(ret_scores, dim=0)

    return ret_seq, ret_scores

# pylint: enable=function-redefined
