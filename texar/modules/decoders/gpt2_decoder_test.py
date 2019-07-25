"""
Unit tests for GPT2 decoder.
"""
import os
import unittest

import torch

from texar.modules.decoders import decoder_helpers
from texar.modules.decoders.gpt2_decoder import GPT2Decoder
from texar.modules.decoders.transformer_decoders import TransformerDecoderOutput

manual_test = unittest.skipUnless(
    os.environ.get('TEST_ALL', 0), "Manual test only")


class GPT2DecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.GPT2Decoder`
    """

    @manual_test
    def test_hparams(self):
        r"""Tests the priority of the decoer arch parameter.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "345M",
        }
        decoder = GPT2Decoder(pretrained_model_name="117M",
                              hparams=hparams)
        _ = decoder(inputs)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "117M",
            "decoder": {
                "num_blocks": 6
            }
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "decoder": {
                "num_blocks": 6
            },
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 6)

        # case 4: using default hparams
        decoder = GPT2Decoder()
        _ = decoder(inputs)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)

    @manual_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: GPT2 117M
        decoder = GPT2Decoder()
        _ = decoder(inputs)
        self.assertEqual(len(decoder.trainable_variables), 1 + 1 + 12 * 26 + 2)

        # case 2: GPT2 345M
        hparams = {
            "pretrained_model_name": "345M"
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs)
        self.assertEqual(len(decoder.trainable_variables), 1 + 1 + 24 * 26 + 2)

        # case 3: self-designed GPT2
        hparams = {
            "decoder": {
                "num_blocks": 6,
            },
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs)
        self.assertEqual(len(decoder.trainable_variables), 1 + 1 + 6 * 26 + 2)

    def test_decode_train(self):
        r"""Tests train_greedy.
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.train()

        max_time = 8
        batch_size = 16
        inputs = torch.randint(50257, (batch_size, max_time), dtype=torch.int64)
        outputs = decoder(inputs)

        self.assertEqual(outputs.logits.shape, torch.Size([batch_size,
                                                           max_time,
                                                           50257]))
        self.assertEqual(outputs.sample_id.shape, torch.Size([batch_size,
                                                              max_time]))

    def test_decode_infer_greedy(self):
        r"""Tests train_greedy
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((16,), 1, dtype=torch.int64)
        end_token = 2
        max_decoding_length = 16

        embedding_fn = lambda x, y: (
                decoder.word_embedder(x) + decoder.position_embedder(y))

        helper = decoder_helpers.GreedyEmbeddingHelper(
            embedding_fn, start_tokens, end_token)

        outputs, length = decoder(
            helper=helper,
            max_decoding_length=max_decoding_length)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_decode_infer_sample(self):
        r"""Tests infer_sample
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((16,), 1, dtype=torch.int64)
        end_token = 2
        max_decoding_length = 16

        embedding_fn = lambda x, y: (
                decoder.word_embedder(x) + decoder.position_embedder(y))

        helper = decoder_helpers.SampleEmbeddingHelper(
            embedding_fn, start_tokens, end_token)

        outputs, length = decoder(
            helper=helper,
            max_decoding_length=max_decoding_length)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_beam_search(self):
        r"""Tests beam_search
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((16,), 1, dtype=torch.int64)
        end_token = 2
        max_decoding_length = 16

        embedding_fn = lambda x, y: (
                decoder.word_embedder(x) + decoder.position_embedder(y))

        outputs = decoder(
            embedding=embedding_fn,
            start_tokens=start_tokens,
            beam_width=5,
            end_token=end_token,
            max_decoding_length=max_decoding_length)

        self.assertEqual(outputs['log_prob'].shape,
                         torch.Size([16, 5]))
        self.assertEqual(outputs['sample_id'].shape,
                         torch.Size([16, 16, 5]))

    def test_greedy_embedding_helper(self):
        r"""Tests with tf.contrib.seq2seq.GreedyEmbeddingHelper
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((16,), 1, dtype=torch.int64)
        end_token = 2
        max_decoding_length = 16

        embedding_fn = lambda x, y: (
                decoder.word_embedder(x) + decoder.position_embedder(y))

        helper = decoder_helpers.GreedyEmbeddingHelper(
            embedding_fn, start_tokens, end_token)

        outputs, length = decoder(
            helper=helper,
            max_decoding_length=max_decoding_length)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_topk_embedding_helper(self):
        r"""Tests TopKSampleEmbeddingHelper
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)
        decoder.eval()

        start_tokens = torch.full((16,), 1, dtype=torch.int64)
        end_token = 2
        max_decoding_length = 16

        embedding_fn = lambda x, y: (
                decoder.word_embedder(x) + decoder.position_embedder(y))

        helper = decoder_helpers.TopKSampleEmbeddingHelper(
            embedding=embedding_fn,
            start_tokens=start_tokens,
            end_token=end_token,
            top_k=40,
            softmax_temperature=0.7)

        outputs, length = decoder(
            max_decoding_length=max_decoding_length,
            helper=helper)

        self.assertIsInstance(outputs, TransformerDecoderOutput)


if __name__ == "__main__":
    unittest.main()
