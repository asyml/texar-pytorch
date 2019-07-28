"""
Unit tests for beam search.
"""

import unittest

import torch
import numpy as np

from texar.torch.utils import beam_search


class BeamSearchTest(unittest.TestCase):
    r"""Tests beam_search.
    """

    def testShapes(self):
        batch_size = 2
        beam_size = 3
        vocab_size = 4
        decode_length = 10

        initial_ids = torch.tensor([0, 0], dtype=torch.int64)

        def symbols_to_logits(_):
            # Just return random logits
            return torch.rand(batch_size * beam_size, vocab_size)

        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=0.0,
            eos_id=1)

        self.assertEqual(final_ids.shape[1], beam_size)
        self.assertEqual(final_probs.shape, torch.Size([batch_size, beam_size]))

    def testComputeTopkScoresAndSeq(self):
        batch_size = 2
        beam_size = 3

        sequences = torch.tensor([[[2, 3], [4, 5], [6, 7], [19, 20]],
                                  [[8, 9], [10, 11], [12, 13], [80, 17]]],
                                 dtype=torch.int64)

        scores = torch.tensor([[-0.1, -2.5, 0., -1.5],
                               [-100., -5., -0.00789, -1.34]])

        flags = torch.tensor([[True, False, False, True],
                              [False, False, False, True]])

        topk_seq, topk_scores, topk_flags, _ = (
            beam_search.compute_topk_scores_and_seq(sequences=sequences,
                                                    scores=scores,
                                                    scores_to_gather=scores,
                                                    flags=flags,
                                                    beam_size=beam_size,
                                                    batch_size=batch_size))

        exp_seq = [[[6, 7], [2, 3], [19, 20]], [[12, 13], [80, 17], [10, 11]]]
        exp_scores = [[0., -0.1, -1.5], [-0.00789, -1.34, -5.]]
        exp_flags = [[False, True, True], [False, True, False]]

        self.assertEqual(topk_seq.tolist(), exp_seq)
        topk_scores = topk_scores.tolist()
        for i in range(2):
            for j in range(3):
                self.assertAlmostEqual(topk_scores[i][j], exp_scores[i][j])
        self.assertEqual(topk_flags.tolist(), exp_flags)

    def testGreedyBatchOne(self):
        batch_size = 1
        beam_size = 1
        vocab_size = 2
        decode_length = 3

        initial_ids = torch.tensor([0] * batch_size, dtype=torch.int64)

        # Test that beam search finds the most probable sequence.
        # These probabilities represent the following search
        #
        #               G0 (0)
        #                  / \
        #                /     \
        #              /         \
        #            /             \
        #         0(0.7)          1(0.3)
        #           / \
        #          /   \
        #         /     \
        #     0(0.4) 1(0.6)
        #        /\
        #       /  \
        #      /    \
        #    0(0.5) 1(0.5)
        # and the following decoding probabilities
        # 0000 - 0.7 * 0.4  * 0.1
        # 0001 - 0.7 * 0.4  * 0.9
        # 001 - 0.7 * 0.6 (Best)
        # 01 = 0.3
        #
        # 001 is the most likely sequence under these probabilities.
        probabilities = torch.tensor([[[0.7, 0.3]], [[0.4, 0.6]], [[0.5, 0.5]]])

        def symbols_to_logits(ids):
            pos = ids.shape[1]
            logits = torch.log(probabilities[pos - 1, :]).type(torch.float)
            return logits

        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=0.0,
            eos_id=1)

        exp_ids = [[[0, 0, 1]]]
        exp_probs = [[0.7 * 0.6]]

        self.assertEqual(final_ids.tolist(), exp_ids)
        self.assertAlmostEqual(np.exp(final_probs).tolist()[0][0],
                               exp_probs[0][0])

    def testNotGreedyBeamTwoWithStopEarly(self):
        batch_size = 1
        beam_size = 2
        vocab_size = 3
        decode_length = 10

        initial_ids = torch.tensor([0] * batch_size, dtype=torch.int64)
        probabilities = torch.tensor([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                      [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                      [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]])

        def symbols_to_logits(ids):
            pos = ids.shape[1]
            logits = torch.log(probabilities[pos - 1, :]).type(torch.float)
            return logits

        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=0.0,
            eos_id=1,
            stop_early=True)  # default value, but just to make this explicit

        # given stop_early = True, the only 'assurance' is w.r.t. the first beam
        # (i.e., other beams may not even be completed)
        # so, we check only the first beam
        first_beam = final_ids[:, 0]
        first_probs = final_probs[:, 0]

        exp_ids = [[0, 2, 1]]
        exp_probs = [0.8 * 0.5]

        self.assertEqual(first_beam.tolist(), exp_ids)
        self.assertAlmostEqual(np.exp(first_probs).tolist()[0], exp_probs[0])

    def testNotGreedyBeamTwoWithoutStopEarly(self):
        batch_size = 1
        beam_size = 2
        vocab_size = 3
        decode_length = 3

        initial_ids = torch.tensor([0] * batch_size, dtype=torch.int64)
        probabilities = torch.tensor([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                      [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                      [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]])

        def symbols_to_logits(ids):
            pos = ids.shape[1]
            logits = torch.log(probabilities[pos - 1, :]).type(torch.float)
            return logits

        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=0.0,
            eos_id=1,
            stop_early=False)

        # given stop_early = False, the algorithm will return all the beams
        # so we can test all of them here

        exp_ids = [[[0, 2, 1, 0], [0, 2, 0, 1]]]
        exp_probs = [[0.8 * 0.5, 0.8 * 0.4 * 0.9]]

        self.assertEqual(final_ids.tolist(), exp_ids)
        self.assertAlmostEqual(np.exp(final_probs).tolist()[0][0],
                               exp_probs[0][0])
        self.assertAlmostEqual(np.exp(final_probs).tolist()[0][1],
                               exp_probs[0][1])

    def testGreedyWithCornerCase(self):
        batch_size = 1
        beam_size = 1
        vocab_size = 3
        decode_length = 2

        initial_ids = torch.tensor([0] * batch_size, dtype=torch.int64)
        probabilities = torch.tensor([[0.2, 0.1, 0.7], [0.4, 0.1, 0.5]])

        def symbols_to_logits(ids):
            pos = ids.shape[1]
            logits = torch.log(probabilities[pos - 1, :]).type(torch.float)
            return logits

        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=0.0,
            eos_id=1)

        exp_ids = [[[0, 2, 2]]]
        exp_probs = [[0.7 * 0.5]]

        self.assertEqual(final_ids.tolist(), exp_ids)
        self.assertAlmostEqual(np.exp(final_probs).tolist()[0][0],
                               exp_probs[0][0])

    def testNotGreedyBatchTwoBeamTwoWithAlpha(self):
        batch_size = 2
        beam_size = 2
        vocab_size = 3
        decode_length = 3

        initial_ids = torch.tensor([0] * batch_size, dtype=torch.int64)
        # Probabilities for position * batch * beam * vocab
        # Probabilities have been set such that with alpha = 3.5, the less
        # probable but longer sequence will have a better score than the
        # shorter sequence with higher log prob in batch 1, and the order will
        # be reverse in batch 2. That is, the shorter sequence will still have
        # a higher score in spite of the length penalty
        probabilities = torch.tensor([[[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                       [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]],
                                      [[[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                       [[0.3, 0.6, 0.1], [0.2, 0.4, 0.4]]],
                                      [[[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]],
                                       [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]]])

        def symbols_to_logits(ids):
            pos = ids.shape[1]
            logits = torch.log(probabilities[pos - 1, :]).type(torch.float)
            return logits

        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=3.5,
            eos_id=1)

        exp_ids = [[[0, 2, 0, 1], [0, 2, 1, 0]], [[0, 2, 1, 0], [0, 2, 0, 1]]]
        exp_probs = [[np.log(0.8 * 0.4 * 0.9) / (8. / 6.)**3.5,
                      np.log(0.8 * 0.5) / (7. / 6.)**3.5],
                     [np.log(0.8 * 0.6) / (7. / 6.)**3.5,
                      np.log(0.8 * 0.3 * 0.9) / (8. / 6.)**3.5]]

        self.assertEqual(final_ids.tolist(), exp_ids)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(final_probs.tolist()[i][j],
                                       exp_probs[i][j])

    def testNotGreedyBeamTwoWithAlpha(self):
        batch_size = 1
        beam_size = 2
        vocab_size = 3
        decode_length = 3

        initial_ids = torch.tensor([0] * batch_size, dtype=torch.int64)
        # Probabilities for position * batch * beam * vocab
        # Probabilities have been set such that with alpha = 3.5, the less
        # probable but longer sequence will have a better score that the
        # shorter sequence with higher log prob.
        probabilities = torch.tensor([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                      [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                      [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]])

        def symbols_to_logits(ids):
            pos = ids.shape[1]
            logits = torch.log(probabilities[pos - 1, :]).type(torch.float)
            return logits

        # Disable early stopping
        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=3.5,
            eos_id=1)

        exp_ids = [[[0, 2, 0, 1], [0, 2, 1, 0]]]
        exp_probs = [[np.log(0.8 * 0.4 * 0.9) / (8. / 6.)**3.5,
                      np.log(0.8 * 0.5) / (7. / 6.)**3.5]]

        self.assertEqual(final_ids.tolist(), exp_ids)
        self.assertAlmostEqual(final_probs.tolist()[0][0], exp_probs[0][0])
        self.assertAlmostEqual(final_probs.tolist()[0][1], exp_probs[0][1])

    def testStates(self):
        batch_size = 1
        beam_size = 1
        vocab_size = 2
        decode_length = 3

        initial_ids = torch.tensor([0] * batch_size, dtype=torch.int64)
        probabilities = torch.tensor([[[0.7, 0.3]], [[0.4, 0.6]], [[0.5, 0.5]]])

        expected_states = torch.tensor([[[0.]], [[1.]]])

        def symbols_to_logits(ids, states):
            pos = ids.shape[1]
            logits = torch.log(probabilities[pos - 1, :]).type(torch.float)
            states["state"] += 1
            return logits, states

        states = {
            "state": torch.zeros(batch_size, 1),
        }

        final_ids, _ = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=0.0,
            eos_id=1,
            states=states)

    def testStateBeamTwo(self):
        batch_size = 1
        beam_size = 2
        vocab_size = 3
        decode_length = 3

        initial_ids = torch.tensor([0] * batch_size, dtype=torch.int64)
        probabilities = torch.tensor([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                      [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                      [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]])

        # The top beam is always selected so we should see the top beam's state
        # at each position, which is the one that getting 3 added to it each
        # step.
        expected_states = torch.tensor([[[0.], [0.]], [[3.], [3.]], [[6.],
                                                                     [6.]]])

        def symbols_to_logits(ids, states):
            pos = ids.shape[1]
            logits = torch.log(probabilities[pos - 1, :]).type(torch.float)
            states["state"] += torch.tensor([[3.], [7.]])
            return logits, states

        states = {
            "state": torch.zeros(batch_size, 1)
        }

        final_ids, _ = beam_search.beam_search(
            symbols_to_logits_fn=symbols_to_logits,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=decode_length,
            vocab_size=vocab_size,
            alpha=0.0,
            eos_id=1,
            states=states)


if __name__ == "__main__":
    unittest.main()
