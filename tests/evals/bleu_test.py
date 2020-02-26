"""
Unit tests for bleu.
"""

import unittest

import numpy as np

from texar.torch.evals.bleu import sentence_bleu, corpus_bleu


class BLEUTest(unittest.TestCase):
    r"""Tests bleu.
    """

    def _test_sentence_bleu(self, references, hypothesis, lowercase,
                            true_bleu):
        bleu = sentence_bleu(references, hypothesis, lowercase=lowercase)
        self.assertAlmostEqual(bleu, true_bleu, places=0)

    def test_sentence_strings(self):
        r"""Tests hypothesis as strings.
        """
        hypothesis = \
            "this is a test sentence to evaluate the good bleu score . 词"
        references = ["this is a test sentence to evaluate the bleu score ."]
        self._test_sentence_bleu(
            references, hypothesis, lowercase=False, true_bleu=67.03)

    def test_sentence_list(self):
        r"""Tests hypothesis as a list of tokens.
        """
        hypothesis = \
            "this is a test sentence to evaluate the good bleu score . 词"
        hypothesis = hypothesis.split()
        references = ["this is a test sentence to evaluate the bleu score ."]
        references = [references[0].split()]
        self._test_sentence_bleu(
            references, hypothesis, lowercase=False, true_bleu=67.03)

    def test_sentence_multi_references(self):
        r"""Tests multiple references.
        """
        hypothesis = \
            "this is a test sentence to evaluate the good bleu score . 词"
        references = ["this is a test sentence to evaluate the bleu score .",
                      "this is a test sentence to evaluate the good score ."]
        self._test_sentence_bleu(
            references, hypothesis, lowercase=False, true_bleu=76.12)

    def test_sentence_numpy(self):
        r"""Tests with numpy format.
        """
        hypothesis = \
            "this is a test sentence to evaluate the good bleu score . 词"
        hypothesis = np.array(hypothesis.split())
        references = ["this is a test sentence to evaluate the bleu score .",
                      "this is a test sentence to evaluate the good score ."]
        references = np.array([np.array(r.split()) for r in references])
        self._test_sentence_bleu(
            references, hypothesis, lowercase=False, true_bleu=76.12)

    def _test_corpus_bleu(self, list_of_references, hypotheses, lowercase,
                          return_all, true_bleu):
        bleu = corpus_bleu(list_of_references, hypotheses,
                           lowercase=lowercase, return_all=return_all)
        if not return_all:
            self.assertAlmostEqual(bleu, true_bleu, places=0)
        else:
            for ret, true in zip(bleu, true_bleu):
                self.assertAlmostEqual(ret, true, places=0)

    def test_corpus_strings(self):
        r"""Tests corpus level BLEU.
        """
        hypotheses = [
            "this is a test sentence to evaluate the good bleu score . 词",
            "i believe that that the script is 词 perfectly correct ."
        ]
        list_of_references = [
            ["this is a test sentence to evaluate the bleu score .",
             "this is a test sentence to evaluate the good score ."],
            ["i believe that the script is perfectly correct .".split()]
        ]
        self._test_corpus_bleu(list_of_references, hypotheses,
                               False, False, 63.02)

        self._test_corpus_bleu(list_of_references, hypotheses,
                               False, True, [63.02, 87.5, 77.3, 60.0, 38.9])


if __name__ == "__main__":
    unittest.main()
