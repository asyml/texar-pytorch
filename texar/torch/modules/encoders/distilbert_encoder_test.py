"""
Unit tests for DistilBERT encoders.
"""

import unittest

import torch

from texar.torch.modules.encoders.distilbert_encoder import DistilBERTEncoder
from texar.torch.utils.test import pretrained_test


class DistilBERTEncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.DistilBERTEncoder` class.
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        for pretrained_model_name in DistilBERTEncoder.available_checkpoints():
            encoder = DistilBERTEncoder(
                pretrained_model_name=pretrained_model_name)
            _, _ = encoder(self.inputs)

    @pretrained_test
    def test_hparams(self):
        r"""Tests the priority of the encoder arch parameter.
        """
        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "distilbert-base-uncased-squad",
        }
        encoder = DistilBERTEncoder(
            pretrained_model_name="distilbert-base-uncased",
            hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)
        _, _ = encoder(self.inputs)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "distilbert-base-uncased-squad",
            "encoder": {
                "num_blocks": 12,
            }
        }
        encoder = DistilBERTEncoder(hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)
        _, _ = encoder(self.inputs)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 12,
            },
        }
        encoder = DistilBERTEncoder(hparams=hparams)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)
        _, _ = encoder(self.inputs)

        # case 4: using default hparams
        encoder = DistilBERTEncoder()
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)
        _, _ = encoder(self.inputs)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1
        encoder = DistilBERTEncoder()
        self.assertEqual(len(encoder.trainable_variables), 3 + 6 * 16 + 2)
        _, _ = encoder(self.inputs)

        # case 2
        hparams = {
            "pretrained_model_name": "distilbert-base-uncased-squad"
        }
        encoder = DistilBERTEncoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), 3 + 6 * 16 + 2)
        _, _ = encoder(self.inputs)

        # case 3: self-designed bert
        hparams = {
            "encoder": {
                "num_blocks": 12,
            },
            "pretrained_model_name": None,
        }
        encoder = DistilBERTEncoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables), 3 + 12 * 16 + 2)
        _, _ = encoder(self.inputs)

    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        encoder = DistilBERTEncoder(hparams=hparams)

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
        encoder = DistilBERTEncoder(hparams=hparams)

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
