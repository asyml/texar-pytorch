"""
Unit tests for GPT2 decoder.
"""
import unittest

import torch

from texar.torch.modules.decoders import decoder_helpers
from texar.torch.modules.decoders.gpt2_decoder import GPT2Decoder
from texar.torch.modules.decoders.transformer_decoders import \
    TransformerDecoderOutput
from texar.torch.utils.test import pretrained_test


class GPT2DecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.GPT2Decoder`
    """

    def setUp(self) -> None:
        # Use small prime numbers to speedup tests.
        self.batch_size = 2
        self.max_length = 3
        self.beam_width = 5
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_hparams(self):
        r"""Tests the priority of the decoder arch parameters.
        """
        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "gpt2-medium",
        }
        decoder = GPT2Decoder(pretrained_model_name="gpt2-small",
                              hparams=hparams)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)
        _ = decoder(self.inputs)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "gpt2-small",
            "decoder": {
                "num_blocks": 6,
            },
        }
        decoder = GPT2Decoder(hparams=hparams)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)
        _ = decoder(self.inputs)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "decoder": {
                "num_blocks": 6,
            },
        }
        decoder = GPT2Decoder(hparams=hparams)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 6)
        _ = decoder(self.inputs)

        # case 4: using default hparams
        decoder = GPT2Decoder()
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)
        _ = decoder(self.inputs)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        def get_variable_num(n_layers: int) -> int:
            return 1 + 1 + n_layers * 26 + 2

        # case 1: GPT2 small
        decoder = GPT2Decoder()
        self.assertEqual(len(decoder.trainable_variables), get_variable_num(12))
        _ = decoder(self.inputs)

        # case 2: GPT2 medium
        hparams = {
            "pretrained_model_name": "gpt2-medium",
        }
        decoder = GPT2Decoder(hparams=hparams)
        self.assertEqual(len(decoder.trainable_variables), get_variable_num(24))
        _ = decoder(self.inputs)

        # case 2: GPT2 large
        hparams = {
            "pretrained_model_name": "gpt2-large",
        }
        decoder = GPT2Decoder(hparams=hparams)
        self.assertEqual(len(decoder.trainable_variables), get_variable_num(36))
        _ = decoder(self.inputs)

        # case 3: self-designed GPT2
        hparams = {
            "pretrained_model_name": None,
            "decoder": {
                "num_blocks": 6,
            },
        }
        decoder = GPT2Decoder(hparams=hparams)
        self.assertEqual(len(decoder.trainable_variables), get_variable_num(6))
        _ = decoder(self.inputs)

    def test_decode_train(self):
        r"""Tests train_greedy.
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.train()

        inputs = torch.randint(50257, (self.batch_size, self.max_length))
        outputs = decoder(inputs)

        self.assertEqual(outputs.logits.shape,
                         torch.Size([self.batch_size, self.max_length, 50257]))
        self.assertEqual(outputs.sample_id.shape,
                         torch.Size([self.batch_size, self.max_length]))

    def test_decode_infer_greedy(self):
        r"""Tests train_greedy
        """
        hparams = {
            "pretrained_model_name": None,
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((self.batch_size,), 1, dtype=torch.int64)
        end_token = 2

        helper = decoder_helpers.GreedyEmbeddingHelper(start_tokens, end_token)

        outputs, length = decoder(
            helper=helper, max_decoding_length=self.max_length)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_decode_infer_sample(self):
        r"""Tests infer_sample
        """
        hparams = {
            "pretrained_model_name": None,
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((self.batch_size,), 1, dtype=torch.int64)
        end_token = 2

        helper = decoder_helpers.SampleEmbeddingHelper(start_tokens, end_token)

        outputs, length = decoder(
            helper=helper, max_decoding_length=self.max_length)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_beam_search(self):
        r"""Tests beam_search
        """
        hparams = {
            "pretrained_model_name": None,
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((self.batch_size,), 1, dtype=torch.int64)
        end_token = 2

        outputs = decoder(
            start_tokens=start_tokens, beam_width=self.beam_width,
            end_token=end_token, max_decoding_length=self.max_length)

        self.assertEqual(
            outputs['log_prob'].shape,
            torch.Size([self.batch_size, self.beam_width]))
        self.assertEqual(
            outputs['sample_id'].shape,
            torch.Size([self.batch_size, self.max_length, self.beam_width]))

    def test_greedy_embedding_helper(self):
        r"""Tests with tf.contrib.seq2seq.GreedyEmbeddingHelper
        """
        hparams = {
            "pretrained_model_name": None,
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((self.batch_size,), 1, dtype=torch.int64)
        end_token = 2

        helper = decoder_helpers.GreedyEmbeddingHelper(start_tokens, end_token)

        outputs, length = decoder(
            helper=helper, max_decoding_length=self.max_length)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_topk_embedding_helper(self):
        r"""Tests TopKSampleEmbeddingHelper
        """
        hparams = {
            "pretrained_model_name": None,
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((self.batch_size,), 1, dtype=torch.int64)
        end_token = 2

        helper = decoder_helpers.TopKSampleEmbeddingHelper(
            start_tokens=start_tokens, end_token=end_token,
            top_k=40, softmax_temperature=0.7)

        outputs, length = decoder(
            helper=helper, max_decoding_length=self.max_length)

        self.assertIsInstance(outputs, TransformerDecoderOutput)


if __name__ == "__main__":
    unittest.main()
