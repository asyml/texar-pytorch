"""
Unit tests for GPT2 encoder.
"""
import unittest

import torch
from texar.torch.modules.encoders.gpt2_encoder import GPT2Encoder
from texar.torch.utils.test import pretrained_test


class GPT2EncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.GPT2Encoder` class.
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        for pretrained_model_name in GPT2Encoder.available_checkpoints():
            encoder = GPT2Encoder(pretrained_model_name=pretrained_model_name)
            _ = encoder(self.inputs)

    @pretrained_test
    def test_hparams(self):
        r"""Tests the priority of the encoder arch parameter.
        """
        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "gpt2-medium",
        }
        encoder = GPT2Encoder(pretrained_model_name="gpt2-small",
                              hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)
        _ = encoder(self.inputs)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "gpt2-small",
            "encoder": {
                "num_blocks": 6,
            }
        }
        encoder = GPT2Encoder(hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)
        _ = encoder(self.inputs)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6,
            }
        }
        encoder = GPT2Encoder(hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)
        _ = encoder(self.inputs)

        # case 4: using default hparams
        encoder = GPT2Encoder()
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)
        _ = encoder(self.inputs)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """

        def get_variable_num(n_layers: int) -> int:
            return 1 + 1 + n_layers * 16 + 2

        # case 1: GPT2 small
        encoder = GPT2Encoder()
        self.assertEqual(len(encoder.trainable_variables), get_variable_num(12))
        _ = encoder(self.inputs)

        # case 2: GPT2 medium
        hparams = {
            "pretrained_model_name": "gpt2-medium",
        }
        encoder = GPT2Encoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), get_variable_num(24))
        _ = encoder(self.inputs)

        # case 2: GPT2 large
        hparams = {
            "pretrained_model_name": "gpt2-large",
        }
        encoder = GPT2Encoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), get_variable_num(36))
        _ = encoder(self.inputs)

        # case 3: self-designed GPT2
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6,
            },
        }
        encoder = GPT2Encoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), get_variable_num(6))
        _ = encoder(self.inputs)

    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1: GPT2 small
        hparams = {
            "pretrained_model_name": None,
        }
        encoder = GPT2Encoder(hparams=hparams)

        inputs = torch.randint(30521, (self.batch_size, self.max_length))
        outputs = encoder(inputs)

        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, encoder.output_size]))

        # case 2: self-designed GPT2
        hparams = {
            'pretrained_model_name': None,
            'embed': {
                'dim': 96,
            },
            'position_embed': {
                'dim': 96,
            },

            'encoder': {
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
                                'bias': True,
                            },
                            'type': 'Linear',
                        },
                        {"type": "GPTGELU"},
                        {
                            'kwargs': {
                                'in_features': 96 * 4,
                                'out_features': 96,
                                'bias': True,
                            },
                            'type': 'Linear',
                        }
                    ]
                },
            }
        }
        encoder = GPT2Encoder(hparams=hparams)

        outputs = encoder(inputs)
        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, encoder.output_size]))

    def test_soft_ids(self):
        r"""Tests soft ids.
        """
        hparams = {
            "pretrained_model_name": None,
        }
        encoder = GPT2Encoder(hparams=hparams)

        inputs = torch.rand(self.batch_size, self.max_length, 50257)
        outputs = encoder(inputs)

        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, encoder.output_size]))


if __name__ == "__main__":
    unittest.main()
