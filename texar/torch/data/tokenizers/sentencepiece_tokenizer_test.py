"""
Unit tests for SentencePiece tokenizer.
"""

import unittest

import os
import pickle
import tempfile

from texar.torch.data.data_utils import maybe_download
from texar.torch.data.tokenizers.sentencepiece_tokenizer import \
    SentencePieceTokenizer


class SentencePieceTokenizerTest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.SAMPLE_VOCAB = maybe_download(
            'https://github.com/google/sentencepiece/blob/master/'
            'python/test/test_model.model?raw=true', self.tmp_dir.name)

        self.tokenizer = SentencePieceTokenizer.load(self.SAMPLE_VOCAB)

        self.tokenizer.save(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load(self):
        tokenizer = SentencePieceTokenizer.load(self.tmp_dir.name)

        self.assertEqual(1000, len(tokenizer))
        self.assertEqual(0, tokenizer.map_token_to_id('<unk>'))
        self.assertEqual(1, tokenizer.map_token_to_id('<s>'))
        self.assertEqual(2, tokenizer.map_token_to_id('</s>'))
        self.assertEqual('<unk>', tokenizer.map_id_to_token(0))
        self.assertEqual('<s>', tokenizer.map_id_to_token(1))
        self.assertEqual('</s>', tokenizer.map_id_to_token(2))
        for i in range(len(tokenizer)):
            token = tokenizer.map_id_to_token(i)
            self.assertEqual(i, tokenizer.map_token_to_id(token))

    def test_roundtrip(self):
        tokenizer = SentencePieceTokenizer.load(self.tmp_dir.name)

        text = 'I saw a girl with a telescope.'
        ids = tokenizer.map_text_to_id(text)
        tokens = tokenizer.map_text_to_token(text)

        self.assertEqual(text, tokenizer.map_id_to_text(ids))
        self.assertEqual(text, tokenizer.map_token_to_text(tokens))

    def test_train(self):
        tmp_dir = tempfile.TemporaryDirectory()
        TEXT_FILE = maybe_download(
            'https://github.com/google/sentencepiece/blob/master/'
            'data/botchan.txt?raw=true', tmp_dir.name)

        hparams = {
            "vocab_file": None,
            "text_file": TEXT_FILE,
            "vocab_size": 1000,
        }
        tokenizer = SentencePieceTokenizer(hparams=hparams)

        with open(TEXT_FILE, 'r', encoding='utf-8') as file:
            for line in file:
                tokenizer.map_token_to_text(tokenizer.map_text_to_token(line))
                tokenizer.map_id_to_text(tokenizer.map_text_to_id(line))

    def test_pickle(self):
        tokenizer = SentencePieceTokenizer.load(self.tmp_dir.name)
        self.assertIsNotNone(tokenizer)

        text = u"Munich and Berlin are nice cities"
        subwords = tokenizer.map_text_to_token(text)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, u"tokenizer.bin")
            with open(filename, "wb") as f:
                pickle.dump(tokenizer, f)
            with open(filename, "rb") as f:
                tokenizer_new = pickle.load(f)

        subwords_loaded = tokenizer_new.map_text_to_token(text)

        self.assertListEqual(subwords, subwords_loaded)

    def test_save_load(self):
        tokenizer = SentencePieceTokenizer.load(self.tmp_dir.name)

        before_tokens = tokenizer.map_text_to_id(
            u"He is very happy, UNwant\u00E9d,running")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save(tmpdirname)
            tokenizer = tokenizer.load(tmpdirname)

        after_tokens = tokenizer.map_text_to_id(
            u"He is very happy, UNwant\u00E9d,running")
        self.assertListEqual(before_tokens, after_tokens)

    def test_add_tokens(self):
        tokenizer = SentencePieceTokenizer.load(self.tmp_dir.name)

        vocab_size = tokenizer.vocab_size
        all_size = len(tokenizer)

        self.assertNotEqual(vocab_size, 0)
        self.assertEqual(vocab_size, all_size)

        new_toks = ["aaaaabbbbbb", "cccccccccdddddddd"]
        added_toks = tokenizer.add_tokens(new_toks)
        vocab_size_2 = tokenizer.vocab_size
        all_size_2 = len(tokenizer)

        self.assertNotEqual(vocab_size_2, 0)
        self.assertEqual(vocab_size, vocab_size_2)
        self.assertEqual(added_toks, len(new_toks))
        self.assertEqual(all_size_2, all_size + len(new_toks))

        tokens = tokenizer.map_text_to_id("aaaaabbbbbb low cccccccccdddddddd l")
        self.assertGreaterEqual(len(tokens), 4)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

        new_toks_2 = {'eos_token': ">>>>|||<||<<|<<",
                      'pad_token': "<<<<<|||>|>>>>|>"}
        added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
        vocab_size_3 = tokenizer.vocab_size
        all_size_3 = len(tokenizer)

        self.assertNotEqual(vocab_size_3, 0)
        self.assertEqual(vocab_size, vocab_size_3)
        self.assertEqual(added_toks_2, len(new_toks_2))
        self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

        tokens = tokenizer.map_text_to_id(
            ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd "
            "<<<<<|||>|>>>>|> l")

        self.assertGreaterEqual(len(tokens), 6)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[0], tokens[1])
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokens[-3])
        self.assertEqual(tokens[0],
                         tokenizer.map_token_to_id(tokenizer.eos_token))
        self.assertEqual(tokens[-2],
                         tokenizer.map_token_to_id(tokenizer.pad_token))


if __name__ == "__main__":
    unittest.main()
