"""
Unit tests for XLNet encoder.
"""
import unittest

import torch

from texar.torch.modules.encoders.xlnet_encoder import XLNetEncoder
from texar.torch.utils.test import pretrained_test


class XLNetEncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.XLNetEncoder` class.
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        for pretrained_model_name in XLNetEncoder.available_checkpoints():
            encoder = XLNetEncoder(pretrained_model_name=pretrained_model_name)
            _ = encoder(self.inputs)

    @pretrained_test
    def test_hparams(self):
        r"""Tests the priority of the encoder arch parameter.
        """
        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        encoder = XLNetEncoder(pretrained_model_name="xlnet-base-cased",
                               hparams=hparams)
        self.assertEqual(encoder.hparams.num_layers, 12)
        _, _ = encoder(self.inputs)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
            "num_layers": 6,
        }
        encoder = XLNetEncoder(hparams=hparams)
        self.assertEqual(encoder.hparams.num_layers, 24)
        _, _ = encoder(self.inputs)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "num_layers": 6,
        }
        encoder = XLNetEncoder(hparams=hparams)
        self.assertEqual(encoder.hparams.num_layers, 6)
        _, _ = encoder(self.inputs)

        # case 4: using default hparams
        encoder = XLNetEncoder()
        self.assertEqual(encoder.hparams.num_layers, 12)
        _, _ = encoder(self.inputs)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1: xlnet base
        encoder = XLNetEncoder()
        self.assertEqual(len(encoder.trainable_variables), 182)
        _, _ = encoder(self.inputs)

        # Case 2: xlnet large
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        encoder = XLNetEncoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), 362)
        _, _ = encoder(self.inputs)

        # case 3: self-designed bert
        hparams = {
            "num_layers": 6,
            "pretrained_model_name": None,
        }
        encoder = XLNetEncoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), 92)
        _, _ = encoder(self.inputs)

    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1: xlnet base
        hparams = {
            "pretrained_model_name": None,
        }
        encoder = XLNetEncoder(hparams=hparams)

        inputs = torch.randint(32000, (self.batch_size, self.max_length))
        outputs, new_memory = encoder(inputs)

        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, encoder.output_size]))
        self.assertEqual(new_memory, None)

        # case 2: self-designed xlnet
        hparams = {
            'pretrained_model_name': None,
            'untie_r': True,
            'num_layers': 6,
            'mem_len': 0,
            'reuse_len': 0,
            'num_heads': 8,
            'hidden_dim': 32,
            'head_dim': 64,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'use_segments': True,
            'ffn_inner_dim': 256,
            'activation': 'gelu',
            'vocab_size': 32000,
            'max_seq_length': 128,
            'initializer': None,
            'name': "xlnet_encoder",
        }
        encoder = XLNetEncoder(hparams=hparams)
        outputs, new_memory = encoder(inputs)

        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, encoder.output_size]))
        self.assertEqual(new_memory, None)

    def test_soft_ids(self):
        r"""Tests soft ids.
        """
        hparams = {
            "pretrained_model_name": None,
        }
        encoder = XLNetEncoder(hparams=hparams)

        inputs = torch.rand(self.batch_size, self.max_length, 32000)
        outputs, new_memory = encoder(inputs)

        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, encoder.output_size]))
        self.assertEqual(new_memory, None)


if __name__ == "__main__":
    unittest.main()
