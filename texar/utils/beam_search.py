# Adapted from the The Tensor2Tensor's implementation.
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
    `https://github.com/tensorflow/tensor2tensor/`
    `blob/master/tensor2tensor/utils/beam_search.py`
"""
from typing import Optional, Dict
import torch

from texar.utils import map_structure

# Default value for INF
INF = 1.0 * 1e7


def gather_nd(params, indices):

    assert len(indices.size()) == 3
    orig_size = params.size()
    index = indices[:, :, 1].view(-1) + indices[:, :, 0].view(-1) * orig_size[1]
    ret = torch.index_select(
        params.view(-1, *params.size()[2:]), dim=0, index=index
    )
    ret = ret.view(orig_size[0], indices.size(1), *orig_size[2:])

    return ret


def _merge_beam_dim(tensor: torch.tensor):
    """Reshapes first two dimensions in to single dimension.
    Args:
        tensor: Tensor to reshape of shape [A, B, ...]
    Returns:
        Reshaped tensor of shape [A*B, ...]
    """
    shape = list(tensor.size())
    shape[0] *= shape[1]  # batch -> batch * beam_size
    shape.pop(1)  # Remove beam dim
    return tensor.view(shape)


def _unmerge_beam_dim(tensor: torch.tensor, batch_size: int, beam_size: int):
    """Reshapes first dimension back to [batch_size, beam_size].
    Args:
        tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
        batch_size: int, original batch size.
        beam_size: int, original beam size.
    Returns:
        Reshaped tensor of shape [batch_size, beam_size, ...]
    """
    shape = list(tensor.size())
    new_shape = [batch_size] + [beam_size] + shape[1:]
    return tensor.view(new_shape)


def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.
    Args:
        tensor: tensor to tile [batch_size, ...]
        beam_size: How much to tile the tensor by.
    Returns:
        Tiled tensor [batch_size, beam_size, ...]
    """
    tensor = torch.unsqueeze(tensor, dim=1)
    tile_dims = [1] * len(tensor.size())
    tile_dims[1] = beam_size

    return tensor.repeat(tile_dims)


def log_prob_from_logits(logits):
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def compute_batch_indices(batch_size: int, beam_size: int):
    """Computes the i'th coordinate that contains the batch index for
    gathers.
    Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    batch the beam item is in. This will create the i of the i,j coordinate
    needed for the gather.
    Args:
        batch_size: Batch size
        beam_size: Size of the beam.
    Returns:
        batch_pos: [batch_size, beam_size] tensor of ids
    """
    batch_pos = torch.arange(batch_size)
    batch_pos = batch_pos.view(-1, 1).expand(batch_size, beam_size)
    return batch_pos


def compute_topk_scores_and_seq(
    sequences,
    scores,
    scores_to_gather,
    flags,
    beam_size,
    batch_size,
    states_to_gather=None,
):
    """Given sequences and scores, will gather the top k=beam size
    sequences.
    This function is used to grow alive, and finished. It takes sequences,
    scores, and flags, and returns the top k from sequence
    scores_to_gather, and flags based on the values in scores.
    This method permits easy introspection using tfdbg. It adds three
    named ops that are prefixed by `prefix`:
        - _topk_seq: the tensor for topk_seq returned by this method.
        - _topk_flags: the tensor for topk_finished_flags returned by this
            method.
        - _topk_scores: the tensor for tokp_gathered_scores returned by
            this method.
    Args:
        sequences: Tensor of sequences that we need to gather from.
            [batch_size, beam_size, seq_length]
        scores: Tensor of scores for each sequence in sequences.
            [batch_size, beam_size]. We will use these to compute the topk.
        scores_to_gather: Tensor of scores for each sequence in sequences.
            [batch_size, beam_size]. We will return the gathered scores
            from here.
            Scores to gather is different from scores because for
            grow_alive, we will need to return log_probs, while for
            grow_finished, we will need to return the length penalized
            scors.
        flags: Tensor of bools for sequences that say whether a sequence
            has reached EOS or not
        beam_size: int
        batch_size: int
        states_to_gather: dict (possibly nested) of decoding states.
    Returns:
        Tuple of
        (topk_seq [batch_size, beam_size, decode_length],
         topk_gathered_scores [batch_size, beam_size],
         topk_finished_flags[batch_size, beam_size])
    """
    # by default topk is for the last dimension
    _, topk_indexes = torch.topk(scores, k=beam_size)
    # The next three steps are to create coordinates for torch.gather_nd to
    # pull out the topk sequences from sequences based on scores.
    # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    # batch the beam item is in. This will create the i of the i,j
    # coordinate needed for the gather
    batch_pos = compute_batch_indices(batch_size, beam_size)
    batch_pos = batch_pos.to(device=topk_indexes.device)
    # top coordinates will give us the actual coordinates to do the gather.
    # stacking will create a tensor of dimension batch * beam * 2, where
    # the last dimension contains the i,j gathering coordinates.
    top_coordinates = torch.stack([batch_pos, topk_indexes], dim=2)

    # Gather up the highest scoring sequences.    For each operation
    # added, give it a concrete name to simplify observing these
    # operations with tfdbg. Clients can capture these tensors by watching
    # these node names.

    topk_seq = gather_nd(sequences, top_coordinates)
    topk_flags = gather_nd(flags, top_coordinates)
    topk_gathered_scores = gather_nd(scores_to_gather, top_coordinates)
    if states_to_gather:
        topk_gathered_states = map_structure(
            lambda state: gather_nd(state, top_coordinates), states_to_gather
        )
    else:
        topk_gathered_states = states_to_gather
    return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


def beam_search(
    symbols_to_logits_fn,
    initial_ids,
    beam_size,
    decode_length,
    vocab_size,
    alpha: float,
    eos_id,
    states=None,
    stop_early=True,
):
    """Beam search with length penalties.
    Requires a function that can take the currently decoded sybmols and
    return the logits for the next symbol. The implementation is inspired
    by https://arxiv.org/abs/1609.08144.
    When running, the beam search steps can be visualized by using tfdbg to
    watch the operations generating the output ids for each beam step.
    These operations have the pattern:
        (alive|finished)_topk_(seq,scores)
    Operations marked `alive` represent the new beam sequences that will be
    processed in the next step.    Operations marked `finished` represent
    the completed beam sequences, which may be padded with 0s if no beams
    finished.
    Operations marked `seq` store the full beam sequence for the time step.
    Operations marked `scores` store the sequence's final log scores.
    The beam search steps will be processed sequentially in order, so when
    capturing observed from these operations, tensors, clients can make
    assumptions about which step is being recorded.
    WARNING: Assumes 2nd dimension of tensors in `states` and not
    invariant, this means that the shape of the 2nd dimension of these
    tensors will not be available (i.e. set to None) inside
    symbols_to_logits_fn.
    Args:
        symbols_to_logits_fn: Interface to the model, to provide logits.
            Should take [batch_size, decoded_ids] and return
            [batch_size, vocab_size]
        initial_ids: Ids to start off the decoding, this will be the first
            thing handed to symbols_to_logits_fn (after expanding to beam size)
            [batch_size]
        beam_size: Size of the beam.
        decode_length: Number of steps to decode for.
        vocab_size: Size of the vocab, must equal the size of the logits
            returned by symbols_to_logits_fn
        alpha: alpha for length penalty.
        states: dict (possibly nested) of decoding states.
        eos_id: ID for end of sentence.
        stop_early: a boolean - stop once best sequence is provably
            determined.
    Returns:
        Tuple of
        (decoded beams [batch_size, beam_size, decode_length]
         decoding probablities [batch_size, beam_size])
    """

    batch_size = initial_ids.size()[0]

    # Assume initial_ids are prob 1.0
    initial_log_probs = torch.tensor(
        [[0.0] + [-float("inf")] * (beam_size - 1)]
    )  # [1, beam_size]
    initial_log_probs = initial_log_probs.to(device=initial_ids.device)
    # Expand to beam_size (batch_size, beam_size)
    alive_log_probs = initial_log_probs.repeat([batch_size, 1])

    # Expand each batch and state to beam_size
    alive_seq = _expand_to_beam_size(initial_ids, beam_size)
    alive_seq = torch.unsqueeze(alive_seq, dim=2)
    # (batch_size, beam_size, 1)

    if states:
        states = map_structure(
            lambda state: _expand_to_beam_size(state, beam_size), states
        )
    else:
        states = {}

    # Finished will keep track of all the sequences that have finished so
    # far
    # Finished log probs will be negative infinity in the beginning
    # finished_flags will keep track of booleans
    finished_seq = torch.zeros(alive_seq.size(), dtype=torch.long)
    # Setting the scores of the initial to negative infinity.
    finished_scores = torch.ones([batch_size, beam_size]) * -INF
    finished_flags = torch.zeros([batch_size, beam_size]).byte()

    finished_seq = finished_seq.to(device=initial_ids.device)
    finished_scores = finished_scores.to(device=initial_ids.device)
    finished_flags = finished_flags.to(device=initial_ids.device)

    def grow_finished(
        finished_seq,
        finished_scores,
        finished_flags,
        curr_seq,
        curr_scores,
        curr_finished,
    ):
        """Given sequences and scores, will gather the top k=beam size
        sequences.
        Args:
            finished_seq: Current finished sequences.
                [batch_size, beam_size, current_decoded_length]
            finished_scores: scores for each of these sequences.
                [batch_size, beam_size]
            finished_flags: finished bools for each of these sequences.
                [batch_size, beam_size]
            curr_seq: current topk sequence that has been grown by one
                position.
                [batch_size, beam_size, current_decoded_length]
            curr_scores: scores for each of these sequences. [batch_size,
                beam_size]
            curr_finished: Finished flags for each of these sequences.
                [batch_size, beam_size]
        Returns:
            Tuple of
                (Topk sequences based on scores,
                 log probs of these sequences,
                 Finished flags of these sequences)
        """
        # First append a column of 0'ids to finished to make the same
        # length with finished scores
        _appended = torch.zeros([batch_size, beam_size, 1], dtype=torch.long)
        _appended = _appended.to(device=finished_seq.device)
        finished_seq = torch.cat([finished_seq, _appended], dim=2)

        # Set the scores of the unfinished seq in curr_seq to large
        # negative values
        curr_scores = curr_scores + (1.0 - curr_finished.float()) * -INF
        # concatenating the sequences and scores along beam axis
        curr_finished_seq = torch.cat([finished_seq, curr_seq], dim=1)
        curr_finished_scores = torch.cat([finished_scores, curr_scores], dim=1)
        curr_finished_flags = torch.cat([finished_flags, curr_finished], dim=1)
        return compute_topk_scores_and_seq(
            curr_finished_seq,
            curr_finished_scores,
            curr_finished_scores,
            curr_finished_flags,
            beam_size,
            batch_size,
        )

    def grow_alive(
        curr_seq, curr_scores, curr_log_probs, curr_finished, states
    ):
        """Given sequences and scores, will gather the top k=beam size
        sequences.
        Args:
            curr_seq: current topk sequence that has been grown by one
                position.
                [batch_size, beam_size, i+1]
            curr_scores: scores for each of these sequences. [batch_size,
                beam_size]
            curr_log_probs: log probs for each of these sequences.
                [batch_size, beam_size]
            curr_finished: Finished flags for each of these sequences.
                [batch_size, beam_size]
            states: dict (possibly nested) of decoding states.
        Returns:
            Tuple of
                (Topk sequences based on scores,
                 log probs of these sequences,
                 Finished flags of these sequences)
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
        i, alive_seq, alive_log_probs: torch.Tensor, states: Optional[Dict]
    ):
        r"""Inner beam seach loop.
        This function takes the current alive sequences, and grows them to
        topk sequences where k = 2*beam. We use 2*beam because, we could
        have beam_size number of sequences that might hit <EOS> and there
        will be no alive sequences to continue. With 2*beam_size, this
        will not happen. This relies on the assumption the vocab size is >
        beam size. If this is true, we'll have at least beam_size non
        <EOS> extensions if we extract the next top 2*beam words.
        Length penalty is given by = (5+len(decode)/6) ^ -\alpha.
        Pls refer to https://arxiv.org/abs/1609.08144.
        Args:
            i: loop index
            alive_seq: Topk sequences decoded so far [batch_size,
                beam_size, i+1]
            alive_log_probs: probabilities of these sequences.
                [batch_size, beam_size]
            states: dict (possibly nested) of decoding states.
        Returns:
            Tuple of
                (Topk sequences extended by the next word,
                 The log probs of these sequences,
                 The scores with length penalty of these sequences,
                 Flags indicating which of these sequences have finished
                 decoding, dict of transformed decoding states)
        """
        # Get the logits for all the possible next symbols
        flat_ids = alive_seq.view(batch_size * beam_size, -1)

        # (batch_size * beam_size, decoded_length)
        if states:
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

        # Convert logits to normalized log probs
        candidate_log_probs = log_prob_from_logits(logits)

        # Multiply the probabilites by the current probabilites of the
        # beam.
        # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
        log_probs = candidate_log_probs + alive_log_probs.unsqueeze(dim=2)

        length_penalty = ((5.0 + float(i + 1)) / 6.0) ** alpha

        curr_scores = log_probs / length_penalty
        # Flatten out (beam_size, vocab_size) probs in to a list of
        # possibilites
        flat_curr_scores = curr_scores.view(-1, beam_size * vocab_size)

        topk_scores, topk_ids = torch.topk(flat_curr_scores, k=beam_size * 2)
        # Recovering the log probs because we will need to send them back
        topk_log_probs = topk_scores * length_penalty

        # Work out what beam the top probs are in.
        topk_beam_index = topk_ids / vocab_size
        topk_ids %= vocab_size  # Unflatten the ids

        # The next three steps are to create coordinates for torch.gather_nd
        # to pull out the correct seqences from id's that we need to grow.
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

        if states:
            states = map_structure(
                lambda state: gather_nd(state, topk_coordinates), states
            )

        # Append the most probable alive
        topk_seq = torch.cat([topk_seq, topk_ids.unsqueeze(dim=2)], dim=2)

        topk_finished = topk_ids == eos_id

        return topk_seq, topk_log_probs, topk_scores, topk_finished, states

    def inner_loop(
        i,
        alive_seq,
        alive_log_probs,
        finished_seq,
        finished_scores,
        finished_flags,
        states,
    ):
        """Inner beam seach loop.
        There are three groups of tensors, alive, finished, and topk.
        The alive group contains information about the current alive
        sequences. The topk group contains information about alive + topk
        current decoded words the finished group contains information
        about finished sentences, that is, the ones that have decoded to
        <EOS>. These are what we return.
        The general beam search algorithm is as follows:
        While we haven't terminated (pls look at termination condition)
            1. Grow the current alive to get beam*2 topk sequences
            2. Among the topk, keep the top beam_size ones that haven't
            reached EOS into alive
            3. Among the topk, keep the top beam_size ones have reached
            EOS into finished
        Repeat
        To make things simple with using fixed size tensors, we will end
        up inserting unfinished sequences into finished in the beginning.
        To stop that we add -ve INF to the score of the unfinished
        sequence so that when a true finished sequence does appear, it
        will have a higher score than all the unfinished ones.
        Args:
            i: loop index
            alive_seq: Topk sequences decoded so far [batch_size,
                beam_size, i+1]
            alive_log_probs: probabilities of the beams. [batch_size,
                beam_size]
            finished_seq: Current finished sequences.
                [batch_size, beam_size, i+1]
            finished_scores: scores for each of these sequences.
                [batch_size, beam_size]
            finished_flags: finished bools for each of these sequences.
                [batch_size, beam_size]
            states: dict (possibly nested) of decoding states.
        Returns:
            Tuple of
                (Incremented loop index
                 New alive sequences,
                 Log probs of the alive sequences,
                 New finished sequences,
                 Scores of the new finished sequences,
                 Flags inidicating which sequence in finished as reached
                 EOS,
                 dict of final decoding states)
        """

        # Each inner loop, we carry out three steps:
        # 1. Get the current topk items.
        # 2. Extract the ones that have finished and haven't finished
        # 3. Recompute the contents of finished based on scores.
        topk_seq, topk_log_probs, topk_scores, topk_finished,\
                states = grow_topk(i, alive_seq, alive_log_probs, states)

        alive_seq, alive_log_probs, _, states = grow_alive(
            topk_seq, topk_scores, topk_log_probs, topk_finished, states
        )
        finished_seq, finished_scores, finished_flags, _ = grow_finished(
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
        i,
        unused_alive_seq,
        alive_log_probs,
        unused_finished_seq,
        finished_scores,
        finished_in_finished: torch.ByteTensor,
        unused_states,
    ):
        """Checking termination condition.
        We terminate when we decoded up to decode_length or the lowest
        scoring item in finished has a greater score that the higest prob
        item in alive divided by the max length penalty
        Args:
            i: loop index
            alive_log_probs: probabilities of the beams. [batch_size,
                beam_size]
            finished_scores: scores for each of these sequences.
                [batch_size, beam_size]
            finished_in_finished: finished boolean tensors for each of these
                sequences. [batch_size, beam_size]
        Returns:
            Bool.
        """
        if not stop_early:
            return i < decode_length
        max_length_penalty = ((5.0 + float(decode_length)) / 6.0) ** alpha
        # The best possible score of the most likley alive sequence
        lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

        # Now to compute the lowest score of a finished sequence in
        # finished
        # If the sequence isn't finished, we multiply it's score by 0.
        # since scores are all -ve, taking the min will give us the score
        # of the lowest finished item.
        lowest_score_of_fininshed_in_finished = torch.min(
            finished_scores * finished_in_finished.float(), dim=1
        ).values

        # If none of the sequences have finished, then the min will be 0
        # and we have to replace it by -ve INF if it is. The score of any
        # seq in alive will be much higher than -ve INF and the
        # termination condition will not be met.
        lowest_score_of_fininshed_in_finished = (
            lowest_score_of_fininshed_in_finished
            + (1.0 - finished_in_finished.any(dim=1).float()) * -INF
        )

        bound_is_met = (
            (lowest_score_of_fininshed_in_finished > lower_bound_alive_scores)
            .all()
            .item()
        )

        ret = (i < decode_length) & (~bound_is_met)

        return ret

    step = 0
    while _is_finished(
        step,
        alive_seq,
        alive_log_probs,
        finished_seq,
        finished_scores,
        finished_flags,
        states,
    ):
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
