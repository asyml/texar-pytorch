"""
Unit tests for Transformer decoder.
"""
import unittest

import torch

from texar.modules import GreedyEmbeddingHelper
from texar.modules.decoders.transformer_decoders import TransformerDecoder
from texar.modules.decoders.transformer_decoders import TransformerDecoderOutput


# pylint: disable=too-many-instance-attributes


class TransformerDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.TransformerDecoder`
    """

    def setUp(self):
        self._vocab_size = 15
        self._batch_size = 6
        self._max_time = 10
        self._emb_dim = 512
        self._max_decode_len = 32
        self._inputs = torch.rand(
            self._batch_size, self._max_time, self._emb_dim, dtype=torch.float)
        self._memory = torch.rand(
            self._batch_size, self._max_time, self._emb_dim, dtype=torch.float)
        self._memory_sequence_length = torch.randint(
            self._max_time, (self._batch_size,), dtype=torch.long)
        self._embedding = torch.rand(
            self._vocab_size, self._emb_dim, dtype=torch.float)
        self._start_tokens = torch.full(
            (self._batch_size,), 1, dtype=torch.long)
        self._end_token = 2
        self.max_decoding_length = self._max_time

        _context = [[3, 4, 5, 2, 0], [4, 3, 5, 7, 2]]
        _context_length = [4, 5]
        self._context = torch.tensor(_context)
        self._context_length = torch.tensor(_context_length)

    def test_train(self):
        """Tests train_greedy
        """
        decoder = TransformerDecoder(embedding=self._embedding)
        decoder.train()
        # 6 blocks
        # -self multihead_attention: 4 dense without bias + 2 layer norm vars
        # -encdec multihead_attention: 4 dense without bias + 2 layer norm vars
        # -poswise_network: Dense with bias, Dense with bias + 2 layer norm vars
        # 2 layer norm vars
        outputs = decoder(memory=self._memory,
                          memory_sequence_length=self._memory_sequence_length,
                          memory_attention_bias=None,
                          inputs=self._inputs,
                          decoding_strategy='train_greedy')
        print(decoder)
        for name, _ in decoder.named_parameters():
            print(name)
        self.assertEqual(len(decoder.trainable_variables), 110)
        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_infer_greedy(self):
        """Tests train_greedy
        """
        decoder = TransformerDecoder(embedding=self._embedding)
        decoder.eval()
        outputs, length = decoder(
            memory=self._memory,
            memory_sequence_length=self._memory_sequence_length,
            memory_attention_bias=None,
            inputs=None,
            decoding_strategy='infer_greedy',
            start_tokens=self._start_tokens,
            end_token=self._end_token,
            max_decoding_length=self._max_decode_len)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_infer_greedy_with_context_without_memory(self):
        """Tests train_greedy with context
        """
        decoder = TransformerDecoder(embedding=self._embedding)
        decoder.eval()
        outputs, length = decoder(
            memory=None,
            memory_sequence_length=None,
            memory_attention_bias=None,
            inputs=None,
            decoding_strategy='infer_greedy',
            context=self._context,
            context_sequence_length=self._context_length,
            end_token=self._end_token,
            max_decoding_length=self._max_decode_len)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_infer_sample(self):
        """Tests infer_sample
        """
        decoder = TransformerDecoder(embedding=self._embedding)
        decoder.eval()
        outputs, length = decoder(
            memory=self._memory,
            memory_sequence_length=self._memory_sequence_length,
            memory_attention_bias=None,
            inputs=None,
            decoding_strategy='infer_sample',
            start_tokens=self._start_tokens,
            end_token=self._end_token,
            max_decoding_length=self._max_decode_len)

        self.assertIsInstance(outputs, TransformerDecoderOutput)

    def test_beam_search(self):
        """Tests beam_search
        """
        return
        decoder = TransformerDecoder(embedding=self._embedding)
        decoder.eval()
        outputs = decoder(
            memory=self._memory,
            memory_sequence_length=self._memory_sequence_length,
            memory_attention_bias=None,
            inputs=None,
            beam_width=5,
            start_tokens=self._start_tokens,
            end_token=self._end_token,
            max_decoding_length=self._max_decode_len)

        self.assertEqual(outputs['log_prob'].shape,
                         (self._batch_size, 5))
        self.assertEqual(outputs['sample_id'].shape,
                         (self._batch_size, self._max_decode_len, 5))

    def test_greedy_embedding_helper(self):
        """Tests with tf.contrib.seq2seq.GreedyEmbeddingHelper
        """
        decoder = TransformerDecoder(embedding=self._embedding)
        decoder.eval()
        helper = GreedyEmbeddingHelper(
            self._embedding, self._start_tokens, self._end_token)
        outputs, length = decoder(
            memory=self._memory,
            memory_sequence_length=self._memory_sequence_length,
            memory_attention_bias=None,
            helper=helper,
            max_decoding_length=self._max_decode_len)

        self.assertIsInstance(outputs, TransformerDecoderOutput)


if __name__ == "__main__":
    unittest.main()
