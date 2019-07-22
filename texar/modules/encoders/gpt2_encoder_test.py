"""
Unit tests for GPT2 encoder.
"""

import unittest

import torch

from texar.modules.encoders.gpt2_encoder import GPT2Encoder


class GPT2EncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.GPT2Encoder` class.
    """

    def test_ci(self):
        r"""Tests for CI."""
        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)

        hparams = {
            "pretrained_model_name": None,
        }
        encoder = GPT2Encoder(hparams=hparams)
        outputs = encoder(inputs)
        outputs_dim = encoder.hparams.decoder.dim
        self.assertEqual(outputs.shape, torch.Size([batch_size,
                                                    max_time,
                                                    outputs_dim]))

    @unittest.skip("Manual test only")
    def test_hparams(self):
        r"""Tests the priority of the encoder arch parameter.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "345M",
        }
        encoder = GPT2Encoder(pretrained_model_name="117M",
                              hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(encoder.hparams.decoder.num_blocks, 12)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "117M",
            "decoder": {
                "num_blocks": 6
            }
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(encoder.hparams.decoder.num_blocks, 12)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "decoder": {
                "num_blocks": 6
            },
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(encoder.hparams.decoder.num_blocks, 6)

        # case 4: using default hparams
        encoder = GPT2Encoder()
        _ = encoder(inputs)
        self.assertEqual(encoder.hparams.decoder.num_blocks, 12)

    @unittest.skip("Manual test only")
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: GPT2 117M
        encoder = GPT2Encoder()
        _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 1 + 1 + 12 * 26 + 2)

        # case 2: GPT2 345M
        hparams = {
            "pretrained_model_name": "345M"
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 1 + 1 + 24 * 26 + 2)

        # case 3: self-designed GPT2
        hparams = {
            "decoder": {
                "num_blocks": 6,
            },
            "pretrained_model_name": None
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 1 + 1 + 6 * 26 + 2)

    @unittest.skip("Manual test only")
    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1: GPT2 117M
        encoder = GPT2Encoder()

        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)
        outputs = encoder(inputs)

        outputs_dim = encoder.hparams.decoder.dim

        self.assertEqual(outputs.shape, torch.Size([batch_size,
                                                    max_time,
                                                    outputs_dim]))

        # case 2: self-designed GPT2
        hparams = {
            'pretrained_model_name': None,
            'embed': {
                'dim': 96
            },
            'position_embed': {
                'dim': 96
            },

            'decoder': {
                'dim': 96,
                'multihead_attention': {
                    'num_units': 96,
                    'output_dim': 96,
                },
                'poswise_feedforward': {
                    'layers': [
                        {
                            'kwargs': {
                                'in_features': 96,
                                'out_features': 96 * 4,
                                'bias': True
                            },
                            'type': 'Linear'
                        },
                        {"type": "GPTGELU"},
                        {
                            'kwargs': {
                                'in_features': 96 * 4,
                                'out_features': 96,
                                'bias': True
                            },
                            'type': 'Linear'
                        }
                    ]
                },
            }
        }
        encoder = GPT2Encoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)
        outputs = encoder(inputs)

        outputs_dim = encoder.hparams.decoder.dim

        self.assertEqual(outputs.shape, torch.Size([batch_size,
                                                    max_time,
                                                    outputs_dim]))


if __name__ == "__main__":
    unittest.main()
