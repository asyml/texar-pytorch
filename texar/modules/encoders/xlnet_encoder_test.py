"""
Unit tests for XLNet encoders.
"""

import unittest

import torch

from texar.modules.encoders.xlnet_encoder import XLNetEncoder


@unittest.skip("Manual test only")
class XLNetEncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.XLNetEncoder` class.
    """

    def test_hparams(self):
        r"""Tests the priority of the encoder arch parameter.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        encoder = XLNetEncoder(pretrained_model_name="xlnet-base-cased",
                               hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.num_layers, 12)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
            "num_layers": 6
        }
        encoder = XLNetEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.num_layers, 24)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "num_layers": 6
        }
        encoder = XLNetEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.num_layers, 6)

        # case 4: using default hparams
        encoder = XLNetEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.num_layers, 12)

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: xlnet base
        encoder = XLNetEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 182)

        # Case 2: xlnet large
        hparams = {
            "pretrained_model_name": "xlnet-large-cased"
        }
        encoder = XLNetEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 362)

        # case 3: self-designed bert
        hparams = {
            "num_layers": 6,
            "pretrained_model_name": None
        }
        encoder = XLNetEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 92)

    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1: xlnet base
        encoder = XLNetEncoder()

        max_time = 8
        batch_size = 16
        inputs = torch.randint(32000, (batch_size, max_time), dtype=torch.int64)
        outputs, new_memory = encoder(inputs)

        self.assertEqual(outputs.shape, torch.Size([batch_size,
                                                    max_time,
                                                    encoder.output_size]))
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
            'max_seq_len': 128,
            'initializer': None,
            'name': "xlnet_encoder",
        }
        encoder = XLNetEncoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        inputs = torch.randint(32000, (batch_size, max_time), dtype=torch.int64)
        outputs, new_memory = encoder(inputs)

        self.assertEqual(outputs.shape, torch.Size([batch_size,
                                                    max_time,
                                                    encoder.output_size]))
        self.assertEqual(new_memory, None)


if __name__ == "__main__":
    unittest.main()
