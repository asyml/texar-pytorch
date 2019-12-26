"""
Unit tests for T5 encoder-decoder
"""

import unittest

import torch

from texar.torch.modules.encoder_decoders import T5EncoderDecoder
from texar.torch.utils.test import pretrained_test


class T5EncoderDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.T5EncoderDecoder` class.

    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        for pretrained_model_name in T5EncoderDecoder.available_checkpoints():
            if pretrained_model_name in ['T5-11B', 'T5-3B']:
                continue  # Too large to fit

            model = T5EncoderDecoder(
                pretrained_model_name=pretrained_model_name)
            _, _ = model(self.inputs)

    @pretrained_test
    def test_hparams(self):
        r"""Tests the priority of the architecture.
        """
        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "T5-Small",
        }
        t5 = T5EncoderDecoder(pretrained_model_name="T5-Base",
                                 hparams=hparams)
        self.assertEqual(t5.hparams.encoder.num_blocks, 12)
        _, _ = t5(self.inputs)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "T5-Small",
            "encoder": {
                "num_blocks": 16,
            }
        }
        t5 = T5EncoderDecoder(hparams=hparams)
        self.assertEqual(t5.hparams.encoder.num_blocks, 6)
        _, _ = t5(self.inputs)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6,
            },
        }
        t5 = T5EncoderDecoder(hparams=hparams)
        self.assertEqual(t5.hparams.encoder.num_blocks, 6)
        _, _ = t5(self.inputs)

        # case 4: using default hparams
        encoder = T5EncoderDecoder()
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)
        _, _ = encoder(self.inputs)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1: t5 small
        encoder = T5EncoderDecoder(pretrained_model_name="T5-Small")
        self.assertEqual(len(encoder.trainable_variables),
                         13 * 6 + 3 + 8 * 6 + 3)
        _, _ = encoder(self.inputs)

        # case 2: bert large
        hparams = {
            "pretrained_model_name": "T5-Base"
        }
        encoder = T5EncoderDecoder(hparams=hparams)
        self.assertEqual(len(encoder.trainable_variables),
                         13 * 12 + 3 + 8 * 12 + 3)
        _, _ = encoder(self.inputs)

    @pretrained_test
    def test_t5(self):
        r"""Tests pretrained model.
        """
        hparams = {
            "pretrained_model_name": 'T5-Small',
        }
        model = T5EncoderDecoder(hparams=hparams)
        import numpy

        self.inputs = torch.from_numpy(
            numpy.asarray([[8774, 6, 82, 1782, 19, 5295]]))
        self.max_length = 6

        encoder_output, decoder_output = model(self.inputs)

        outputs_dim = model.hparams.encoder.dim
        self.assertEqual(
            decoder_output[0].shape,
            torch.Size([self.inputs.size()[0], self.max_length, outputs_dim]))

        self.assertEqual(
            encoder_output.shape,
            torch.Size([self.inputs.size()[0], self.max_length, outputs_dim]))


if __name__ == "__main__":
    unittest.main()
