# -*- coding: utf-8 -*-
#
"""
Unit tests for bleu_tool.
"""

import unittest

import tempfile

from texar.torch.evals.bleu_transformer import (
    bleu_transformer_tokenize, file_bleu)


class BLEUToolTest(unittest.TestCase):
    r"""Test bleu_tool.
    """

    def _create_temp_file(self, text):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, "w") as f:
            f.write(text)
        return temp_file.name

    def test_bleu_same(self):
        ref = self._create_temp_file("test 1 two 3\nmore tests!")
        hyp = self._create_temp_file("test 1 two 3\nmore tests!")

        uncased_score = file_bleu(ref, hyp, case_sensitive=False)
        cased_score = file_bleu(ref, hyp, case_sensitive=True)
        self.assertEqual(100, uncased_score)
        self.assertEqual(100, cased_score)

    def test_bleu_same_different_case(self):
        ref = self._create_temp_file("Test 1 two 3\nmore tests!")
        hyp = self._create_temp_file("test 1 two 3\nMore tests!")
        uncased_score = file_bleu(ref, hyp, case_sensitive=False)
        cased_score = file_bleu(ref, hyp, case_sensitive=True)
        self.assertEqual(100, uncased_score)
        self.assertLess(cased_score, 100)

    def test_bleu_different(self):
        ref = self._create_temp_file("Testing\nmore tests!")
        hyp = self._create_temp_file("Dog\nCat")
        uncased_score = file_bleu(ref, hyp, case_sensitive=False)
        cased_score = file_bleu(ref, hyp, case_sensitive=True)
        self.assertLess(uncased_score, 100)
        self.assertLess(cased_score, 100)

    def test_bleu_tokenize(self):
        s = "Test0, 1 two, 3"
        tokenized = bleu_transformer_tokenize(s)
        self.assertEqual(["Test0", ",", "1", "two", ",", "3"], tokenized)

    def test_bleu_version(self):
        ref = self._create_temp_file("Test 1 two 3\nmore tests!")
        hyp = self._create_temp_file("test 1 two 3\nMore tests!")
        uncased_score = file_bleu(
            ref, hyp, bleu_version="corpus_bleu", case_sensitive=False)
        cased_score = file_bleu(
            ref, hyp, bleu_version="corpus_bleu", case_sensitive=True)
        self.assertEqual(100, uncased_score)
        self.assertLess(cased_score, 100)

        uncased_score = file_bleu(
            ref, hyp, bleu_version="corpus_bleu_moses", case_sensitive=False)
        cased_score = file_bleu(
            ref, hyp, bleu_version="corpus_bleu_moses", case_sensitive=True)
        self.assertEqual(100, uncased_score)
        self.assertLess(cased_score, 100)


if __name__ == "__main__":
    unittest.main()
