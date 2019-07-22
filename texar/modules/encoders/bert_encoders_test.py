"""
Unit tests for BERT encoders.
"""

import unittest

import torch

from texar.modules.encoders.bert_encoders import BERTEncoder


class BERTEncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.BERTEncoder` class.
    """

    def test_ci(self):
        r"""Tests for CI."""
        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)

        hparams = {
            "pretrained_model_name": None,
        }
        encoder = BERTEncoder(hparams=hparams)
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        pooled_output_dim = encoder.hparams.hidden_size

        self.assertEqual(outputs.shape, torch.Size([batch_size,
                                                    max_time,
                                                    outputs_dim]))
        self.assertEqual(pooled_output.shape, torch.Size([batch_size,
                                                          pooled_output_dim]))

    @unittest.skip("Manual test only")
    def test_hparams(self):
        r"""Tests the priority of the encoder arch parameter.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "bert-large-uncased",
        }
        encoder = BERTEncoder(pretrained_model_name="bert-base-uncased",
                              hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "bert-large-uncased",
            "encoder": {
                "num_blocks": 6
            }
        }
        encoder = BERTEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 24)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6
            },
        }
        encoder = BERTEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)

        # case 4: using default hparams
        encoder = BERTEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)

    @unittest.skip("Manual test only")
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: bert base
        encoder = BERTEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3 + 2 + 12 * 16 + 2)

        # case 2: bert large
        hparams = {
            "pretrained_model_name": "bert-large-uncased"
        }
        encoder = BERTEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3 + 2 + 24 * 16 + 2)

        # case 3: self-designed bert
        hparams = {
            "encoder": {
                "num_blocks": 6,
            },
            "pretrained_model_name": None
        }
        encoder = BERTEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3 + 2 + 6 * 16 + 2)

    @unittest.skip("Manual test only")
    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1: bert base
        encoder = BERTEncoder()

        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        pooled_output_dim = encoder.hparams.hidden_size

        self.assertEqual(outputs.shape, torch.Size([batch_size,
                                                    max_time,
                                                    outputs_dim]))
        self.assertEqual(pooled_output.shape, torch.Size([batch_size,
                                                          pooled_output_dim]))

        # case 2: self-designed bert
        hparams = {
            'pretrained_model_name': None,
            'embed': {
                'dim': 96
            },
            'segment_embed': {
                'dim': 96
            },
            'position_embed': {
                'dim': 96
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
                                'bias': True
                            },
                            'type': 'Linear'
                        },
                        {"type": "BertGELU"},
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
            },
            'hidden_size': 96
        }
        encoder = BERTEncoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        pooled_output_dim = encoder.hparams.hidden_size

        self.assertEqual(outputs.shape, torch.Size([batch_size,
                                                    max_time,
                                                    outputs_dim]))
        self.assertEqual(pooled_output.shape, torch.Size([batch_size,
                                                          pooled_output_dim]))


if __name__ == "__main__":
    unittest.main()
