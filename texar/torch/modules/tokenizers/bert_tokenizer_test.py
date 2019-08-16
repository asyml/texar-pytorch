"""
Unit tests for pre-trained BERT tokenizer.
"""

import unittest

import os
import pickle
import tempfile

from texar.torch.modules.tokenizers.bert_tokenizer import BERTTokenizer
from texar.torch.utils.test import pretrained_test


class BERTTokenizerTest(unittest.TestCase):

    def setUp(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un",
            "runn",
            "##ing", ",", "low", "lowest",
        ]

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.vocab_file = os.path.join(self.tmp_dir.name, 'vocab.txt')
        with open(self.vocab_file, "w", encoding='utf-8') as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        self.tmp_dir.cleanup()

    @pretrained_test
    def test_model_loading(self):
        for pretrained_model_name in BERTTokenizer.available_checkpoints():
            tokenizer = BERTTokenizer(
                pretrained_model_name=pretrained_model_name)
            _ = tokenizer.tokenize(u"UNwant\u00E9d,running")

    def test_tokenize(self):
        tokenizer = BERTTokenizer.load(self.vocab_file)

        tokens = tokenizer.tokenize(u"UNwant\u00E9d,running")
        self.assertListEqual(tokens,
                             ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens),
                             [7, 4, 5, 10, 8, 9])

    def test_pickle(self):
        tokenizer = BERTTokenizer.load(self.vocab_file)
        self.assertIsNotNone(tokenizer)

        text = u"Munich and Berlin are nice cities"
        subwords = tokenizer.tokenize(text)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, u"tokenizer.bin")
            with open(filename, "wb") as f:
                pickle.dump(tokenizer, f)
            with open(filename, "rb") as f:
                tokenizer_new = pickle.load(f)

        subwords_loaded = tokenizer_new.tokenize(text)

        self.assertListEqual(subwords, subwords_loaded)

    def test_save_load(self):
        tokenizer = BERTTokenizer.load(self.vocab_file)

        before_tokens = tokenizer.encode(
            u"He is very happy, UNwant\u00E9d,running")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save(tmpdirname)
            tokenizer = tokenizer.load(tmpdirname)

        after_tokens = tokenizer.encode(
            u"He is very happy, UNwant\u00E9d,running")
        self.assertListEqual(before_tokens, after_tokens)

    def test_pretrained_model_list(self):
        model_list_1 = list(BERTTokenizer._MODEL2URL.keys())
        model_list_2 = list(BERTTokenizer._MAX_INPUT_SIZE.keys())

        self.assertListEqual(model_list_1, model_list_2)

    def test_encode_decode(self):
        tokenizer = BERTTokenizer.load(self.vocab_file)

        input_text = u"UNwant\u00E9d,running"
        output_text = u"unwanted, running"

        tokens = tokenizer.tokenize(input_text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids_2 = tokenizer.encode(input_text)
        self.assertListEqual(ids, ids_2)

        tokens_2 = tokenizer.convert_ids_to_tokens(ids)
        text_2 = tokenizer.decode(ids)

        self.assertEqual(text_2, output_text)

        self.assertNotEqual(len(tokens_2), 0)
        self.assertIsInstance(text_2, str)

    def test_add_tokens(self):
        tokenizer = BERTTokenizer.load(self.vocab_file)

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

        tokens = tokenizer.encode("aaaaabbbbbb low cccccccccdddddddd l")
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

        tokens = tokenizer.encode(
            ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd "
            "<<<<<|||>|>>>>|> l")

        self.assertGreaterEqual(len(tokens), 6)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[0], tokens[1])
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokens[-3])
        self.assertEqual(tokens[0],
                         tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
        self.assertEqual(tokens[-2],
                         tokenizer.convert_tokens_to_ids(tokenizer.pad_token))


if __name__ == "__main__":
    unittest.main()
