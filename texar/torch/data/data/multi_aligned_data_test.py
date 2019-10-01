# -*- coding: utf-8 -*-
#
"""
Unit tests for data related operations.
"""
import copy
import os
import tempfile
import unittest

import numpy as np
import torch

from texar.torch.data.data.data_iterators import DataIterator
from texar.torch.data.data.multi_aligned_data import MultiAlignedData
from texar.torch.data.data.record_data import RecordData


class MultiAlignedDataTest(unittest.TestCase):
    """Tests multi aligned text data class.
    """

    def setUp(self):
        # Create test data
        vocab_list = ['This', 'is', 'a', 'word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        text_0 = ['This is a sentence from source .', '词 词 。 source']
        text_0_file = tempfile.NamedTemporaryFile()
        text_0_file.write('\n'.join(text_0).encode("utf-8"))
        text_0_file.flush()
        self._text_0_file = text_0_file

        text_1 = ['This is a sentence from target .', '词 词 。 target']
        text_1_file = tempfile.NamedTemporaryFile()
        text_1_file.write('\n'.join(text_1).encode("utf-8"))
        text_1_file.flush()
        self._text_1_file = text_1_file

        text_2 = [
            'This is a sentence from dialog . ||| dialog ',
            '词 词 。 ||| 词 dialog']
        text_2_file = tempfile.NamedTemporaryFile()
        text_2_file.write('\n'.join(text_2).encode("utf-8"))
        text_2_file.flush()
        self._text_2_file = text_2_file

        int_3 = [0, 1]
        int_3_file = tempfile.NamedTemporaryFile()
        int_3_file.write(('\n'.join([str(_) for _ in int_3])).encode("utf-8"))
        int_3_file.flush()
        self._int_3_file = int_3_file

        self._record_filepath = os.path.join(
            tempfile.mkdtemp(), 'test.pkl')
        self._feature_types = {
            'number1': ('tf.int64', 'stacked_tensor'),
            'number2': ('tf.int64', 'stacked_tensor'),
            'text': ('tf.string', 'stacked_tensor')
        }

        features = [{
            "number1": 128,
            "number2": 512,
            "text": "This is a sentence for TFRecord 词 词 。"
        },
            {
                "number1": 128,
                "number2": 512,
                "text": "This is a another sentence for TFRecord 词 词 。"
            }]
        # Prepare Validation data
        with RecordData.writer(self._record_filepath,
                               self._feature_types) as writer:
            for feature in features:
                writer.write(feature)

        # Construct database
        self._hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "datasets": [
                {  # dataset 0
                    "files": [self._text_0_file.name],
                    "vocab_file": self._vocab_file.name,
                    "bos_token": "",
                    "data_name": "0"
                },
                {  # dataset 1
                    "files": [self._text_1_file.name],
                    "vocab_share_with": 0,
                    "eos_token": "<TARGET_EOS>",
                    "data_name": "1"
                },
                {  # dataset 2
                    "files": [self._text_2_file.name],
                    "vocab_file": self._vocab_file.name,
                    "processing_share_with": 0,
                    # TODO(avinash) - Add it back once feature is added
                    "variable_utterance": False,
                    "data_name": "2"
                },
                {  # dataset 3
                    "files": self._int_3_file.name,
                    "data_type": "int",
                    "data_name": "label"
                },
                {  # dataset 4
                    "files": self._record_filepath,
                    "feature_types": self._feature_types,
                    "feature_convert_types": {
                        'number2': 'tf.float32',
                    },
                    "num_shards": 2,
                    "shard_id": 1,
                    "data_type": "record",
                    "data_name": "4"
                }
            ]
        }

    def _run_and_test(self, hparams, discard_index=None):
        # Construct database
        text_data = MultiAlignedData(hparams)
        self.assertEqual(
            text_data.vocab(0).size,
            self._vocab_size + len(text_data.vocab(0).special_tokens))

        iterator = DataIterator(text_data)
        for batch in iterator:
            self.assertEqual(set(batch.keys()),
                             set(text_data.list_items()))
            text_0 = batch['0_text']
            text_1 = batch['1_text']
            text_2 = batch['2_text']
            int_3 = batch['label']
            number_1 = batch['4_number1']
            number_2 = batch['4_number2']
            text_4 = batch['4_text']

            for t0, t1, t2, i3, n1, n2, t4 in zip(
                    text_0, text_1, text_2, int_3,
                    number_1, number_2, text_4):

                np.testing.assert_array_equal(t0[:2], t1[1:3])
                np.testing.assert_array_equal(t0[:3], t2[1:4])
                if t0[0].startswith('This'):
                    self.assertEqual(i3, 0)
                else:
                    self.assertEqual(i3, 1)
                self.assertEqual(n1, 128)
                self.assertEqual(n2, 512)
                self.assertIsInstance(n1, torch.Tensor)
                self.assertIsInstance(n2, torch.Tensor)
                self.assertIsInstance(t4, str)

            if discard_index is not None:
                hpms = text_data._hparams.datasets[discard_index]
                max_l = hpms.max_seq_length
                max_l += sum(int(x is not None and x != '')
                             for x in [text_data.vocab(discard_index).bos_token,
                                       text_data.vocab(discard_index).eos_token]
                             )
                for i in range(2):
                    for length in batch[text_data.length_name(i)]:
                        self.assertLessEqual(length, max_l)

                # TODO(avinash): Add this back once variable utterance is added
                # for lengths in batch[text_data.length_name(2)]:
                #    for length in lengths:
                #        self.assertLessEqual(length, max_l)

            for i, hpms in enumerate(text_data._hparams.datasets):
                if hpms.data_type != "text":
                    continue
                max_l = hpms.max_seq_length
                mode = hpms.length_filter_mode
                if max_l is not None and mode == "truncate":
                    max_l += sum(int(x is not None and x != '')
                                 for x in [text_data.vocab(i).bos_token,
                                           text_data.vocab(i).eos_token])
                    for length in batch[text_data.length_name(i)]:
                        self.assertLessEqual(length, max_l)

    def test_default_setting(self):
        """Tests the logics of the text data.
        """
        self._run_and_test(self._hparams)

    def test_length_filter(self):
        """Tests filtering by length.
        """
        hparams = copy.copy(self._hparams)
        hparams["datasets"][0].update(
            {"max_seq_length": 4,
             "length_filter_mode": "discard"})
        hparams["datasets"][1].update(
            {"max_seq_length": 2,
             "length_filter_mode": "truncate"})
        self._run_and_test(hparams, discard_index=0)

    def test_supported_scalar_types(self):
        """Tests scalar types supported in MultiAlignedData."""
        # int64 type
        hparams = copy.copy(self._hparams)
        hparams["datasets"][3].update({
            "data_type": "int64"
        })
        self._run_and_test(hparams)

        # float type
        hparams = copy.copy(self._hparams)
        hparams["datasets"][3].update({
            "data_type": "float"
        })
        self._run_and_test(hparams)

        # float64 type
        hparams = copy.copy(self._hparams)
        hparams["datasets"][3].update({
            "data_type": "float64"
        })
        self._run_and_test(hparams)

        # bool type
        hparams = copy.copy(self._hparams)
        hparams["datasets"][3].update({
            "data_type": "bool"
        })
        self._run_and_test(hparams)

    def test_unsupported_scalar_types(self):
        """Tests if exception is thrown for unsupported types."""
        hparams = copy.copy(self._hparams)
        hparams["datasets"][3].update({
            "data_type": "XYZ"
        })

        with self.assertRaises(ValueError):
            self._run_and_test(hparams)

        hparams = copy.copy(self._hparams)
        hparams["datasets"][3].update({
            "data_type": "str"
        })

        with self.assertRaises(ValueError):
            self._run_and_test(hparams)


if __name__ == "__main__":
    unittest.main()
