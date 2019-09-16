"""
Unit tests for utility functions.
"""

import unittest

import numpy as np
import torch

from texar.torch.utils import utils


class UtilsTest(unittest.TestCase):
    r"""Tests utility functions.
    """

    def test_sequence_mask(self):
        r"""Tests :meth:`texar.torch.utils.sequence_mask`.
        """
        mask1 = utils.sequence_mask([1, 3, 2], 5).numpy()
        expected1 = np.asarray(
            [[True, False, False, False, False],
             [True, True, True, False, False],
             [True, True, False, False, False]]
        )
        np.testing.assert_array_equal(mask1, expected1)

        mask2 = utils.sequence_mask(torch.tensor([[1, 3], [2, 0]]))
        expected2 = np.asarray(
            [[[True, False, False],
              [True, True, True]],
             [[True, True, False],
              [False, False, False]]]
        )
        np.testing.assert_array_equal(mask2, expected2)

    def test_dict_patch(self):
        r"""Tests :meth:`texar.torch.utils.dict_patch`.
        """
        src_dict = {
            "k1": "k1",
            "k_dict_1": {
                "kd1_k1": "kd1_k1",
                "kd1_k2": "kd1_k2"
            },
            "k_dict_2": {
                "kd2_k1": "kd2_k1"
            }
        }
        tgt_dict = {
            "k1": "k1_tgt",
            "k_dict_1": {
                "kd1_k1": "kd1_k1"
            },
            "k_dict_2": "kd2_not_dict"
        }

        patched_dict = utils.dict_patch(tgt_dict, src_dict)
        self.assertEqual(patched_dict["k1"], tgt_dict["k1"])
        self.assertEqual(patched_dict["k_dict_1"], src_dict["k_dict_1"])
        self.assertEqual(patched_dict["k_dict_2"], tgt_dict["k_dict_2"])

    def test_strip_token(self):
        r"""Tests :func:`texar.torch.utils.strip_token`
        """
        str_ = " <PAD>  <PAD>\t  i am <PAD> \t <PAD>  \t"
        self.assertEqual(utils.strip_token(str_, "<PAD>"), "i am")
        self.assertEqual(utils.strip_token(str_, ""),
                         "<PAD> <PAD> i am <PAD> <PAD>")
        self.assertEqual(utils.strip_token([str_], "<PAD>"), ["i am"])
        self.assertEqual(
            utils.strip_token(np.asarray([str_]), "<PAD>"),
            ["i am"])
        self.assertEqual(type(utils.strip_token(np.asarray([str_]), "<PAD>")),
                         np.ndarray)
        self.assertEqual(
            utils.strip_token([[[str_]], ['']], "<PAD>"),
            [[["i am"]], ['']])

        str_ = str_.split()
        self.assertEqual(utils.strip_token(str_, "<PAD>", is_token_list=True),
                         ["i", "am"])
        self.assertEqual(utils.strip_token([str_], "<PAD>", is_token_list=True),
                         [["i", "am"]])

    def test_strip_bos(self):
        r"""Tests :func:`texar.torch.utils.strip_bos`
        """
        str_ = "<BOS> i am"
        self.assertEqual(utils.strip_bos(str_, "<BOS>"), "i am")
        self.assertEqual(utils.strip_bos(str_, ""), "<BOS> i am")
        self.assertEqual(utils.strip_bos([str_], "<BOS>"), ["i am"])

        str_ = str_.split()
        self.assertEqual(utils.strip_bos(str_, "<BOS>", is_token_list=True),
                         ["i", "am"])
        self.assertEqual(utils.strip_bos([str_], "<BOS>", is_token_list=True),
                         [["i", "am"]])

    def test_strip_eos(self):
        r"""Tests :func:`texar.torch.utils.strip_eos`
        """
        str_ = "i am <EOS>"
        self.assertEqual(utils.strip_eos(str_, "<EOS>"), "i am")
        self.assertEqual(utils.strip_eos([str_], "<EOS>"), ["i am"])

        str_ = str_.split()
        self.assertEqual(utils.strip_eos(str_, "<EOS>", is_token_list=True),
                         ["i", "am"])
        self.assertEqual(utils.strip_eos([str_], "<EOS>", is_token_list=True),
                         [["i", "am"]])

    def test_strip_special_tokens(self):
        r"""Test :func:`texar.torch.utils.strip_special_tokens`
        """
        str_ = "<BOS> i am <EOS> <PAD> <PAD>"
        self.assertEqual(utils.strip_special_tokens(str_), "i am")
        self.assertEqual(utils.strip_special_tokens([str_]), ["i am"])

        str_ = str_.split()
        self.assertEqual(utils.strip_special_tokens(str_, is_token_list=True),
                         ["i", "am"])
        self.assertEqual(utils.strip_special_tokens([str_], is_token_list=True),
                         [["i", "am"]])

    def test_str_join(self):
        r"""Tests :func:`texar.torch.utils.str_join`
        """
        tokens = np.ones([2, 2, 3], dtype='str')

        str_ = utils.str_join(tokens)
        np.testing.assert_array_equal(
            str_, np.asarray([['1 1 1', '1 1 1'], ['1 1 1', '1 1 1']]))
        self.assertIsInstance(str_, np.ndarray)

        str_ = utils.str_join(tokens.tolist())
        np.testing.assert_array_equal(
            str_, [['1 1 1', '1 1 1'], ['1 1 1', '1 1 1']])
        self.assertIsInstance(str_, list)

        tokens = [[], ['1', '1']]
        str_ = utils.str_join(tokens)
        np.testing.assert_array_equal(str_, ['', '1 1'])

    def test_uniquify_str(self):
        r"""Tests :func:`texar.torch.utils.uniquify_str`.
        """
        str_set = ['str']
        unique_str = utils.uniquify_str('str', str_set)
        self.assertEqual(unique_str, 'str_1')

        str_set.append('str_1')
        str_set.append('str_2')
        unique_str = utils.uniquify_str('str', str_set)
        self.assertEqual(unique_str, 'str_3')

    def test_sum_tensors(self):

        inputs = [torch.tensor(1), torch.tensor(2)]
        self.assertEqual(utils.sum_tensors(inputs), torch.tensor(3))

        inputs = [torch.tensor(1), None, torch.tensor(2)]
        self.assertEqual(utils.sum_tensors(inputs), torch.tensor(3))

        inputs = [torch.tensor(1), None, None]
        self.assertEqual(utils.sum_tensors(inputs), torch.tensor(1))

        inputs = [None, None, None]
        self.assertEqual(utils.sum_tensors(inputs), None)

    def test_truncate_seq_pair(self):

        tokens_a = [1, 2, 3]
        tokens_b = [4, 5, 6]
        utils.truncate_seq_pair(tokens_a, tokens_b, 4)
        self.assertListEqual(tokens_a, [1, 2])
        self.assertListEqual(tokens_b, [4, 5])

        tokens_a = [1]
        tokens_b = [2, 3, 4, 5]
        utils.truncate_seq_pair(tokens_a, tokens_b, 3)
        self.assertListEqual(tokens_a, [1])
        self.assertListEqual(tokens_b, [2, 3])

    # def test_map_ids_to_strs(self):
    #    """Tests :func:`texar.torch.utils.map_ids_to_strs`.
    #    """
    #    vocab_list = ['word', '词']
    #    vocab_file = tempfile.NamedTemporaryFile()
    #    vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
    #    vocab_file.flush()
    #    vocab = Vocab(vocab_file.name)

    #    text = [['<BOS>', 'word', '词', '<EOS>', '<PAD>'],
    #            ['word', '词', 'word', '词', '<PAD>']]
    #    text = np.asarray(text)
    #    ids = vocab.map_tokens_to_ids_py(text)

    #    ids = ids.tolist()
    #    text_ = utils.map_ids_to_strs(ids, vocab)

    #    self.assertEqual(text_[0], 'word 词')
    #    self.assertEqual(text_[1], 'word 词 word 词')


if __name__ == "__main__":
    unittest.main()
