# -*- coding: utf-8 -*-
#
"""
Unit tests for data related operations.
"""
import tempfile
import copy
import numpy as np

import unittest

from texar.torch.data.data.data_iterators import DataIterator
from texar.torch.data.data.paired_text_data import PairedTextData
from texar.torch.data.vocabulary import SpecialTokens


class PairedTextDataTest(unittest.TestCase):
    """Tests paired text data class.
    """

    def setUp(self):

        # Create test data
        vocab_list = ['This', 'is', 'a', 'word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        src_text = ['This is a sentence from source .', '词 词 。 source']
        src_text_file = tempfile.NamedTemporaryFile()
        src_text_file.write('\n'.join(src_text).encode("utf-8"))
        src_text_file.flush()
        self._src_text_file = src_text_file

        tgt_text = ['This is a sentence from target .', '词 词 。 target']
        tgt_text_file = tempfile.NamedTemporaryFile()
        tgt_text_file.write('\n'.join(tgt_text).encode("utf-8"))
        tgt_text_file.flush()
        self._tgt_text_file = tgt_text_file

        self._hparams = {
            "num_epochs": 1,
            "batch_size": 3,
            "source_dataset": {
                "files": [self._src_text_file.name],
                "vocab_file": self._vocab_file.name,
            },
            "target_dataset": {
                "files": self._tgt_text_file.name,
                "vocab_share": True,
                "eos_token": "<TARGET_EOS>"
            }
        }

        self.src_upper_cased_text = []
        for sent in src_text:
            upper_cased_tokens = sent.upper().split(" ")
            upper_cased_tokens.append(SpecialTokens.EOS)
            self.src_upper_cased_text.append(upper_cased_tokens)

        max_length = max([len(tokens) for tokens in self.src_upper_cased_text])
        self.src_upper_cased_text = [sent + [''] * (max_length - len(sent)) for
                                     sent in self.src_upper_cased_text]

        self.tgt_upper_cased_text = []
        for sent in tgt_text:
            upper_cased_tokens = sent.upper().split(" ")
            upper_cased_tokens.insert(0, "<BOS>")
            upper_cased_tokens.append("<TARGET_EOS>")
            self.tgt_upper_cased_text.append(upper_cased_tokens)

        max_length = max([len(tokens) for tokens in self.tgt_upper_cased_text])
        self.tgt_upper_cased_text = [sent + [''] * (max_length - len(sent)) for
                                     sent in self.tgt_upper_cased_text]

    def _run_and_test(self, hparams, proc_shr=False, test_transform=None,
                      discard_src=False):
        # Construct database
        text_data = PairedTextData(hparams)
        self.assertEqual(text_data.source_vocab.size,
                         self._vocab_size +
                         len(text_data.source_vocab.special_tokens))

        iterator = DataIterator(text_data)
        for data_batch in iterator:
            self.assertEqual(set(data_batch.keys()),
                             set(text_data.list_items()))

            if proc_shr:
                tgt_eos = '<EOS>'
            else:
                tgt_eos = '<TARGET_EOS>'

            # Test matching
            src_text = data_batch['source_text']
            tgt_text = data_batch['target_text']
            if proc_shr:
                for src, tgt in zip(src_text, tgt_text):
                    np.testing.assert_array_equal(src[:3], tgt[:3])
            else:
                for src, tgt in zip(src_text, tgt_text):
                    np.testing.assert_array_equal(src[:3], tgt[1:4])
            self.assertTrue(
                tgt_eos in data_batch['target_text'][0])

            if test_transform:
                for i in range(len(data_batch['source_text'])):
                    text_ = data_batch['source_text'][i]
                    self.assertTrue(text_ in self.src_upper_cased_text)
                for i in range(len(data_batch['target_text'])):
                    text_ = data_batch['target_text'][i]
                    self.assertTrue(text_ in self.tgt_upper_cased_text)

            if discard_src:
                src_hparams = text_data.hparams.source_dataset
                max_l = src_hparams.max_seq_length
                max_l += sum(int(x is not None and x != '')
                             for x in [text_data._src_bos_token,
                                       text_data._tgt_bos_token])
                for l in data_batch["source_length"]:
                    self.assertLessEqual(l, max_l)

    def test_default_setting(self):
        """Tests the logics of the text data.
        """
        self._run_and_test(self._hparams)

    def test_shuffle(self):
        """Tests toggling shuffle.
        """
        hparams = copy.copy(self._hparams)
        hparams["shuffle"] = False
        self._run_and_test(hparams)

    def test_processing_share(self):
        """Tests sharing processing.
        """
        hparams = copy.copy(self._hparams)
        hparams["target_dataset"]["processing_share"] = True
        self._run_and_test(hparams, proc_shr=True)

    def test_other_transformations(self):
        """Tests use of other transformations
        """
        _upper_func = lambda raw_example: [x.upper() for x in raw_example]

        hparams = copy.copy(self._hparams)
        hparams["source_dataset"].update(
            {"other_transformations": [_upper_func]})
        hparams["target_dataset"].update(
            {"other_transformations": [_upper_func]})
        self._run_and_test(hparams, test_transform=True)

    def test_length_filter(self):
        """Tests filtering by length.
        """
        hparams = copy.copy(self._hparams)
        hparams["source_dataset"].update(
            {"max_seq_length": 4,
             "length_filter_mode": "discard"})
        self._run_and_test(hparams, discard_src=True)


if __name__ == "__main__":
    unittest.main()
