"""
Unit tests for RoBERTa encoders.
"""

import unittest

import torch

from texar.torch.modules.encoders.roberta_encoder import RoBERTaEncoder
from texar.torch.utils.test import pretrained_test


class RoBERTaEncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.RoBERTaEncoder` class.
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        for pretrained_model_name in RoBERTaEncoder.available_checkpoints():
            encoder = RoBERTaEncoder(
                pretrained_model_name=pretrained_model_name)
            _, _ = encoder(self.inputs)

    @pretrained_test
    def test_hparams(self):
        r"""Tests the priority of the encoder arch parameter.
        """
        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "roberta-large",
        }
        encoder = RoBERTaEncoder(pretrained_model_name="roberta-base",
                                 hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)
        _, _ = encoder(self.inputs)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "roberta-large",
            "encoder": {
                "num_blocks": 6,
            }
        }
        encoder = RoBERTaEncoder(hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 24)
        _, _ = encoder(self.inputs)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6,
            },
        }
        encoder = RoBERTaEncoder(hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)
        _, _ = encoder(self.inputs)

        # case 4: using default hparams
        encoder = RoBERTaEncoder()
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)
        _, _ = encoder(self.inputs)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1: bert base
        encoder = RoBERTaEncoder()
        self.assertEqual(len(encoder.trainable_variables), 2 + 2 + 12 * 16 + 2)
        _, _ = encoder(self.inputs)

        # case 2: bert large
        hparams = {
            "pretrained_model_name": "roberta-large"
        }
        encoder = RoBERTaEncoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), 2 + 2 + 24 * 16 + 2)
        _, _ = encoder(self.inputs)

        # case 3: self-designed bert
        hparams = {
            "encoder": {
                "num_blocks": 6,
            },
            "pretrained_model_name": None,
        }
        encoder = RoBERTaEncoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), 2 + 2 + 6 * 16 + 2)
        _, _ = encoder(self.inputs)

    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1: bert base
        hparams = {
            "pretrained_model_name": None,
        }
        encoder = RoBERTaEncoder(hparams=hparams)

        inputs = torch.randint(30521, (self.batch_size, self.max_length))
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, outputs_dim]))
        self.assertEqual(
            pooled_output.shape,
            torch.Size([self.batch_size, encoder.output_size]))

        # case 2: self-designed bert
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
                        {"type": "BertGELU"},
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
            },
            'hidden_size': 96,
        }
        encoder = RoBERTaEncoder(hparams=hparams)

        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, outputs_dim]))
        self.assertEqual(
            pooled_output.shape,
            torch.Size([self.batch_size, encoder.output_size]))

    def test_soft_ids(self):
        r"""Tests soft ids.
        """
        hparams = {
            "pretrained_model_name": None,
        }
        encoder = RoBERTaEncoder(hparams=hparams)

        inputs = torch.rand(self.batch_size, self.max_length, 50265)
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        self.assertEqual(
            outputs.shape,
            torch.Size([self.batch_size, self.max_length, outputs_dim]))
        self.assertEqual(
            pooled_output.shape,
            torch.Size([self.batch_size, encoder.output_size]))


if __name__ == "__main__":
    unittest.main()
