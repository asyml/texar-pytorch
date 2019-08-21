"""
Unit tests for pre-trained RoBERTa tokenizer.
"""

import unittest

import json
import os
import pickle
import tempfile

from texar.torch.modules.tokenizers.pretrained_roberta_tokenizer import \
    RoBERTaTokenizer
from texar.torch.utils.test import pretrained_test


class RoBERTaTokenizerTest(unittest.TestCase):

    def setUp(self):
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n",
                 "lo", "low", "er",
                 "low", "lowest", "newer", "wider", "<unk>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.vocab_file = os.path.join(self.tmp_dir.name, 'vocab.json')
        self.merges_file = os.path.join(self.tmp_dir.name, 'merges.txt')

        with open(self.vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.merges_file, "w") as fp:
            fp.write("\n".join(merges))

    def tearDown(self):
        self.tmp_dir.cleanup()

    @pretrained_test
    def test_model_loading(self):
        for pretrained_model_name in \
                RoBERTaTokenizer.available_checkpoints():
            tokenizer = RoBERTaTokenizer(
                pretrained_model_name=pretrained_model_name)
            _ = tokenizer(inputs=u"Munich and Berlin are nice cities",
                          task='text-to-token')

    def test_tokenize(self):
        tokenizer = RoBERTaTokenizer.load(self.tmp_dir.name,
                                          self.special_tokens_map)

        text = "lower"
        bpe_tokens = ["low", "er"]
        tokens = tokenizer(inputs=text, task='text-to-token')
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [13, 12, 17]
        self.assertListEqual(
            tokenizer(inputs=input_tokens, task='token-to-id'),
            input_bpe_tokens)

    def test_pickle(self):
        tokenizer = RoBERTaTokenizer.load(self.tmp_dir.name,
                                          self.special_tokens_map)
        self.assertIsNotNone(tokenizer)

        text = u"Munich and Berlin are nice cities"
        subwords = tokenizer(inputs=text, task='text-to-token')

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, u"tokenizer.bin")
            with open(filename, "wb") as f:
                pickle.dump(tokenizer, f)
            with open(filename, "rb") as f:
                tokenizer_new = pickle.load(f)

        subwords_loaded = tokenizer_new.tokenize(text)

        self.assertListEqual(subwords, subwords_loaded)

    def test_save_load(self):
        tokenizer = RoBERTaTokenizer.load(self.tmp_dir.name,
                                          self.special_tokens_map)

        before_tokens = tokenizer(
            inputs=u"He is very happy, UNwant\u00E9d,running",
            task='text-to-id')

        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save(tmpdirname)
            tokenizer = tokenizer.load(tmpdirname)

        after_tokens = tokenizer(
            inputs=u"He is very happy, UNwant\u00E9d,running",
            task='text-to-id')
        self.assertListEqual(before_tokens, after_tokens)

    def test_pretrained_model_list(self):
        model_list_1 = list(RoBERTaTokenizer._MODEL2URL.keys())
        model_list_2 = list(RoBERTaTokenizer._MAX_INPUT_SIZE.keys())

        self.assertListEqual(model_list_1, model_list_2)

    def test_encode_decode(self):
        tokenizer = RoBERTaTokenizer.load(self.tmp_dir.name,
                                          self.special_tokens_map)

        input_text = u"lower newer"
        output_text = u"lower<unk>newer"

        tokens = tokenizer(inputs=input_text, task='text-to-token')
        ids = tokenizer(inputs=tokens, task='token-to-id')
        ids_2 = tokenizer(inputs=input_text, task='text-to-id')
        self.assertListEqual(ids, ids_2)

        tokens_2 = tokenizer(inputs=ids, task='id-to-token')
        text_2 = tokenizer(inputs=ids, task='id-to-text')

        self.assertEqual(text_2, output_text)

        self.assertNotEqual(len(tokens_2), 0)
        self.assertIsInstance(text_2, str)

    def test_add_tokens(self):
        tokenizer = RoBERTaTokenizer.load(self.tmp_dir.name,
                                          self.special_tokens_map)

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
