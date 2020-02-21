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

    def test_lazy_groups_of(self):
        xs = [1, 2, 3, 4, 5, 6, 7]
        groups = utils.lazy_groups_of(iter(xs), group_size=3)
        assert next(groups) == [1, 2, 3]
        assert next(groups) == [4, 5, 6]
        assert next(groups) == [7]
        with self.assertRaises(StopIteration):
            _ = next(groups)

    def test_sort_batch_by_length(self):
        tensor = torch.rand([5, 7, 9])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 1:, :] = 0
        tensor[3, 5:, :] = 0

        sequence_lengths = torch.LongTensor([3, 4, 1, 5, 7])
        sorted_tensor, sorted_lengths, reverse_indices, _ = \
            utils.sort_batch_by_length(tensor, sequence_lengths)

        # Test sorted indices are padded correctly.
        np.testing.assert_array_equal(sorted_tensor[1, 5:, :].data.numpy(), 0.0)
        np.testing.assert_array_equal(sorted_tensor[2, 4:, :].data.numpy(), 0.0)
        np.testing.assert_array_equal(sorted_tensor[3, 3:, :].data.numpy(), 0.0)
        np.testing.assert_array_equal(sorted_tensor[4, 1:, :].data.numpy(), 0.0)

        assert sorted_lengths.data.equal(torch.LongTensor([7, 5, 4, 3, 1]))

        # Test restoration indices correctly recover the original tensor.
        assert sorted_tensor.index_select(0, reverse_indices).data.equal(
            tensor.data)

    def test_combine_initial_dims(self):
        tensor = torch.randn(4, 10, 20, 17, 5)

        tensor2d = utils.combine_initial_dims(tensor)
        assert list(tensor2d.size()) == [4 * 10 * 20 * 17, 5]

    def test_uncombine_initial_dims(self):
        embedding2d = torch.randn(4 * 10 * 20 * 17 * 5, 12)

        embedding = utils.uncombine_initial_dims(embedding2d,
                                                 torch.Size((4, 10, 20, 17, 5)))
        assert list(embedding.size()) == [4, 10, 20, 17, 5, 12]


if __name__ == "__main__":
    unittest.main()
