import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import texar as tx
from texar.modules.encoders import TransformerEncoder
from texar.modules.decoders import TransformerDecoder

from texar.module_base import ModuleBase


class Transformer(ModuleBase):
    def __init__(self, vocab_size, config_model, config_data):
        ModuleBase.__init__(self)
        self.config_model = config_model
        self.config_data = config_data

        self.modules = {}
        src_word_embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams=config_model.emb
        )
        self.modules["src_word_embedder"] = src_word_embedder

        pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=config_data.max_decoding_length,
            hparams=config_model.position_embedder_hparams,
        )
        self.modules["pos_embedder"] = pos_embedder

        encoder = TransformerEncoder(hparams=config_model.encoder)
        self.modules["encoder"] = encoder
        # The decoder ties the input word embedding with the output logit layer.
        # As the decoder masks out <PAD>'s embedding, which in effect means
        # <PAD> has all-zero embedding, so here we explicitly set <PAD>'s
        # embedding to all-zero.
        tgt_embedding = torch.cat(
            [
                torch.zeros((1, src_word_embedder.dim), dtype=torch.float),
                src_word_embedder.embedding[1:, :],
            ],
            dim=0,
        )
        tgt_embedder = tx.modules.WordEmbedder(tgt_embedding)
        self.modules["tgt_embedder"] = tgt_embedder
        _output_w = torch.transpose(tgt_embedder.embedding, 1, 0)

        decoder = TransformerDecoder(
            vocab_size=vocab_size, output_layer=_output_w, hparams=config_model.decoder
        )
        self.modules["decoder"] = decoder

        self.criterion = LabelSmoothingLoss

    def forward(
        self, encoder_input, decoder_input, labels, learning_rate, is_train_mode
    ):

        if is_train_mode:
            self.train()
            batch_size = encoder_input.size[0]
            # (text sequence length excluding padding)
            encoder_input_length = (1 - (encoder_input == 0).int()).sum(dim=1)
            decoder_input_length = (1 - (decoder_input == 0).int()).sum(dim=1)

            # labels = torch.placeholder(torch.int64, shape=(None, None))
            is_target = (labels == 0).float()

            # Source word embedding
            src_word_embeds = self.modules["src_word_embedder"](encoder_input)
            src_word_embeds = src_word_embeds * self.config_model.hidden_dim ** 0.5

            # Position embedding (shared b/w source and target)

            src_seq_len = (
                torch.ones([batch_size], dtype=torch.int32) * encoder_input.size()[1]
            )
            src_pos_embeds = self.modules["pos_embedder"](sequence_length=src_seq_len)
            src_input_embedding = src_word_embeds + src_pos_embeds

            encoder_output = self.modules["encoder"](
                inputs=src_input_embedding, sequence_length=encoder_input_length
            )

            tgt_word_embeds = self.modules["tgt_embedder"](decoder_input)
            tgt_word_embeds = tgt_word_embeds * self.config_model.hidden_dim ** 0.5

            tgt_seq_len = (
                torch.ones(batch_size, dtype=torch.int) * decoder_input.size()[1]
            )
            tgt_pos_embeds = self.modules["pos_embedder"](sequence_length=tgt_seq_len)

            tgt_input_embedding = tgt_word_embeds + tgt_pos_embeds

            # For training
            outputs = self.modules["decoder"](
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                inputs=tgt_input_embedding,
                decoding_strategy="train_greedy",
            )

            mle_loss = self.criterion(outputs.logits, labels)
            mle_loss = (mle_loss * is_target).sum() / is_target.sum()

            # TODO(haoransh): define how to get train_op here
            train_op = tx.core.get_train_op(
                mle_loss,
                learning_rate=learning_rate,
                global_step=global_step,
                hparams=config_model.opt,
            )

            # torch.summary.scalar('lr', learning_rate)
            # torch.summary.scalar('mle_loss', mle_loss)
            # summary_merged = torch.summary.merge_all()
        else:
            batch_size = encoder_input.size[0]
            # (text sequence length excluding padding)
            encoder_input_length = (1 - (encoder_input == 0).int()).sum(dim=1)
            decoder_input_length = (1 - (decoder_input == 0).int()).sum(dim=1)

            # labels = torch.placeholder(torch.int64, shape=(None, None))
            is_target = (labels == 0).float()

            # Source word embedding
            src_word_embeds = src_word_embedder(encoder_input)
            src_word_embeds = src_word_embeds * config_model.hidden_dim ** 0.5

            # Position embedding (shared b/w source and target)

            src_seq_len = (
                torch.ones([batch_size], dtype=torch.int32) * encoder_input.size()[1]
            )
            src_pos_embeds = pos_embedder(sequence_length=src_seq_len)
            src_input_embedding = src_word_embeds + src_pos_embeds

            encoder_output = encoder(
                inputs=src_input_embedding, sequence_length=encoder_input_length
            )
            start_tokens = [bos_token_id for _ in range(batch_size)]

            def _embedding_fn(x, y):
                return tgt_embedder(x) * config_model.hidden_dim ** 0.5 + pos_embedder(
                    y
                )

            predictions = decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                beam_width=beam_width,
                length_penalty=config_model.length_penalty,
                start_tokens=start_tokens,
                end_token=eos_token_id,
                embedding=_embedding_fn,
                max_decoding_length=config_data.max_decoding_length,
                mode=torch.estimator.ModeKeys.PREDICT,
            )
            # Uses the best sample by beam search
            beam_search_ids = predictions["sample_id"][:, :, 0]


def embedding_to_padding(emb):
    """Calculates the padding mask based on which embeddings are all zero.
    We have hacked symbol_modality to return all-zero embeddings
    for padding.

    Args:
        emb: a Tensor with shape [..., depth].

    Returns:
        a float Tensor with shape [...].
    """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.to_float(tf.equal(emb_sum, 0.0))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=0):
        """
        :param label_smoothing:
        :param tgt_vocab_size:
        :param ignore_index: The index in the vocabulary to ignore
        """
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')