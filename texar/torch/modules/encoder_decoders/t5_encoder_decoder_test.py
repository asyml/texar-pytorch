# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for T5 encoder-decoder
"""

import unittest

import numpy
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
    def test_t5_eval(self):
        r"""Tests pre-trained model and check it generates
        same results everytime.
        """
        hparams = {
            "pretrained_model_name": 'T5-Small',
        }
        model = T5EncoderDecoder(hparams=hparams)
        model.eval()

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

        # Check if these value are same consistently. If not, there is something
        # wrong with the pre-trained model.
        self.assertEqual(
            encoder_output.data[0][3][345].tolist(),
            -0.16204041242599487
        )
        self.assertLess(  # leave some margin for minor stochastic differences
            decoder_output[0].data[0][0][234].tolist() + 0.325570285320282,
            0.000001
        )

    def test_t5(self):
        r"""t5 test.
        """
        hparams = {
           "pretrained_model_name": None,
        }

        t5 = T5EncoderDecoder(hparams=hparams)

        inputs = torch.randint(32128, (self.batch_size, self.max_length))

        encoder_output, decoder_output = t5(inputs)

        outputs_dim = t5.output_size

        self.assertEqual(
            decoder_output[0].shape,
            torch.Size([self.batch_size, self.max_length, outputs_dim]))

        self.assertEqual(
            encoder_output.shape,
            torch.Size([self.batch_size, self.max_length, outputs_dim]))


if __name__ == "__main__":
    unittest.main()
