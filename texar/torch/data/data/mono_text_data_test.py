"""
Unit tests for data related operations.
"""
import copy
import tempfile
import unittest

import numpy as np

from texar.torch.data.data.data_iterators import DataIterator
from texar.torch.data.data.mono_text_data import MonoTextData
from texar.torch.data.vocabulary import SpecialTokens


class MonoTextDataTest(unittest.TestCase):
    r"""Tests text data class.
    """

    def setUp(self):
        # Create test data
        vocab_list = ['word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        text = ['This is a test sentence .', '词 词 。']
        text_file = tempfile.NamedTemporaryFile()
        text_file.write('\n'.join(text).encode("utf-8"))
        text_file.flush()
        self._text_file = text_file

        self._hparams = {
            "num_epochs": 1,
            "batch_size": 3,
            "dataset": {
                "files": self._text_file.name,
                "vocab_file": self._vocab_file.name,
            }
        }

        self.upper_cased_text = []
        for sent in text:
            upper_cased_tokens = sent.upper().split(" ")
            upper_cased_tokens.insert(0, SpecialTokens.BOS)
            upper_cased_tokens.append(SpecialTokens.EOS)
            self.upper_cased_text.append(upper_cased_tokens)

        max_length = max([len(tokens) for tokens in self.upper_cased_text])
        self.upper_cased_text = [sent + [''] * (max_length - len(sent)) for sent
                                 in self.upper_cased_text]

    def _run_and_test(self,
                      hparams,
                      test_batch_size=False,
                      test_transform=False):
        # Construct database
        text_data = MonoTextData(hparams)
        self.assertEqual(text_data.vocab.size,
                         self._vocab_size + len(text_data.vocab.special_tokens))

        iterator = DataIterator(text_data)

        for data_batch in iterator:
            self.assertEqual(set(data_batch.keys()),
                             set(text_data.list_items()))

            if test_batch_size:
                self.assertEqual(len(data_batch['text']), hparams['batch_size'])

            if test_transform:
                for i in range(len(data_batch['text'])):
                    text_ = data_batch['text'][i]
                    self.assertTrue(text_ in self.upper_cased_text)

            max_seq_length = text_data.hparams.dataset.max_seq_length
            mode = text_data.hparams.dataset.length_filter_mode

            max_l = max_seq_length
            if max_seq_length is not None:
                if text_data.hparams.dataset.eos_token != '':
                    max_l += 1
                if text_data.hparams.dataset.bos_token != '':
                    max_l += 1

            if max_seq_length == 6:
                for length in data_batch['length']:
                    self.assertLessEqual(length, max_l)
                if mode == "discard":
                    for length in data_batch['length']:
                        self.assertEqual(length, 5)
                elif mode == "truncate":
                    num_length_6 = 0
                    for length in data_batch['length']:
                        num_length_6 += int(length == 6)
                    self.assertGreater(num_length_6, 0)
                else:
                    raise ValueError("Unknown mode: %s" % mode)

            if text_data.hparams.dataset.pad_to_max_seq_length:
                for x in data_batch['text']:
                    self.assertEqual(len(x), max_l)
                for x in data_batch['text_ids']:
                    self.assertEqual(len(x), max_l)

    def test_default_setting(self):
        r"""Tests the logic of MonoTextData.
        """
        self._run_and_test(self._hparams)

    def test_batching(self):
        r"""Tests different batching.
        """
        # disallow smaller final batch
        hparams = copy.deepcopy(self._hparams)
        hparams.update({"allow_smaller_final_batch": False})
        self._run_and_test(hparams, test_batch_size=True)

    @unittest.skip("bucketing is not yet implemented")
    def test_bucketing(self):
        r"""Tests bucketing.
        """
        hparams = copy.deepcopy(self._hparams)
        hparams.update({
            "bucket_boundaries": [7],
            "bucket_batch_sizes": [6, 4]})

        text_data = MonoTextData(hparams)
        iterator = DataIterator(text_data)

        hparams.update({
            "bucket_boundaries": [7],
            "bucket_batch_sizes": [7, 7],
            "allow_smaller_final_batch": False})

        text_data_1 = MonoTextData(hparams)
        iterator_1 = DataIterator(text_data_1)

        for data_batch, data_batch_1 in zip(iterator, iterator_1):
            length = data_batch['length'][0]
            if length < 7:
                last_batch_size = hparams['num_epochs'] % 6
                self.assertTrue(
                    len(data_batch['text']) == 6 or
                    len(data_batch['text']) == last_batch_size)
            else:
                last_batch_size = hparams['num_epochs'] % 4
                self.assertTrue(
                    len(data_batch['text']) == 4 or
                    len(data_batch['text']) == last_batch_size)

            self.assertEqual(len(data_batch_1['text']), 7)

    def test_shuffle(self):
        r"""Tests different shuffling strategies.
        """
        hparams = copy.deepcopy(self._hparams)
        hparams.update({
            "shard_and_shuffle": True,
            "shuffle_buffer_size": 1})
        self._run_and_test(hparams)

    def test_prefetch(self):
        r"""Tests prefetching.
        """
        hparams = copy.deepcopy(self._hparams)
        hparams.update({"prefetch_buffer_size": 2})
        self._run_and_test(hparams)

    def test_other_transformations(self):
        r"""Tests use of other transformations
        """

        upper_func = lambda x: [w.upper() for w in x]

        hparams = copy.deepcopy(self._hparams)
        hparams["dataset"].update(
            {"other_transformations": [upper_func]})
        self._run_and_test(hparams, test_transform=True)

    def test_list_items(self):
        r"""Tests the item names of the output data.
        """
        text_data = MonoTextData(self._hparams)
        self.assertSetEqual(set(text_data.list_items()),
                            {"text", "text_ids", "length"})

        hparams = copy.deepcopy(self._hparams)
        hparams["dataset"]["data_name"] = "data"
        text_data = MonoTextData(hparams)
        self.assertSetEqual(set(text_data.list_items()),
                            {"data_text", "data_text_ids", "data_length"})

    def test_length_discard(self):
        r"""Tests discard length seq.
        """
        hparams = copy.deepcopy(self._hparams)
        hparams["dataset"].update({"max_seq_length": 4,
                                   "length_filter_mode": "discard"})
        self._run_and_test(hparams)

    def test_length_truncate(self):
        r"""Tests truncation.
        """
        hparams = copy.deepcopy(self._hparams)
        hparams["dataset"].update({"max_seq_length": 4,
                                   "length_filter_mode": "truncate"})
        hparams["shuffle"] = False
        hparams["allow_smaller_final_batch"] = False
        self._run_and_test(hparams)

    def test_pad_to_max_length(self):
        r"""Tests padding.
        """
        hparams = copy.deepcopy(self._hparams)
        hparams["dataset"].update({"max_seq_length": 10,
                                   "length_filter_mode": "truncate",
                                   "pad_to_max_seq_length": True})
        self._run_and_test(hparams)


@unittest.skip("Skipping until Variable Utterance is implemented")
class VarUttMonoTextDataTest(unittest.TestCase):
    r"""Tests variable utterance text data class.
    """

    def setUp(self):
        # Create test data
        vocab_list = ['word', 'sentence', '词', 'response', 'dialog', '1', '2']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        text = [
            'This is a dialog 1 sentence . ||| This is a dialog 1 sentence . '
            '||| This is yet another dialog 1 sentence .',  # //
            'This is a dialog 2 sentence . ||| '
            'This is also a dialog 2 sentence . ',  # //
            '词 词 词 ||| word',  # //
            'This This',  # //
            '1 1 1 ||| 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ||| 1 1 1 ||| 2'
        ]
        text_file = tempfile.NamedTemporaryFile()
        text_file.write('\n'.join(text).encode("utf-8"))
        text_file.flush()
        self._text_file = text_file

        self._hparams = {
            "num_epochs": 1,
            "batch_size": 3,
            "shuffle": False,
            "dataset": {
                "files": self._text_file.name,
                "vocab_file": self._vocab_file.name,
                "variable_utterance": True,
                "max_utterance_cnt": 3,
                "max_seq_length": 10
            }
        }

    def _run_and_test(self, hparams):
        # Construct database
        text_data = MonoTextData(hparams)
        self.assertEqual(text_data.vocab.size,
                         self._vocab_size + len(text_data.vocab.special_tokens))

        iterator = DataIterator(text_data)

        for data_batch in iterator:
            # Run the logics
            self.assertEqual(set(data_batch.keys()),
                             set(text_data.list_items()))

            # Test utterance count
            utt_ind = np.sum(data_batch["text_ids"], 2) != 0
            utt_cnt = np.sum(utt_ind, 1)
            self.assertListEqual(
                data_batch[text_data.utterance_cnt_name].tolist(),
                utt_cnt.tolist())

            if text_data.hparams.dataset.pad_to_max_seq_length:
                max_l = text_data.hparams.dataset.max_seq_length
                max_l += text_data._decoder.added_length
                for x in data_batch['text']:
                    for xx in x:
                        self.assertEqual(len(xx), max_l)
                for x in data_batch['text_ids']:
                    for xx in x:
                        self.assertEqual(len(xx), max_l)

    def test_default_setting(self):
        r"""Tests the logics of the text data.
        """
        self._run_and_test(self._hparams)

    def test_pad_to_max_length(self):
        r"""Tests padding.
        """
        hparams = copy.copy(self._hparams)
        hparams["dataset"].update({"max_seq_length": 20,
                                   "length_filter_mode": "truncate",
                                   "pad_to_max_seq_length": True})
        self._run_and_test(hparams)


if __name__ == "__main__":
    unittest.main()
