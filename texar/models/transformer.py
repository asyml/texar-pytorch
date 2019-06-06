import pickle
from typing import Optional

import torch
from torch import nn

import texar as tx
from texar.losses.mle_losses import sequence_softmax_cross_entropy
from texar.module_base import ModuleBase
from texar.modules.decoders import TransformerDecoder
from texar.modules.encoders import TransformerEncoder


class Transformer(ModuleBase):
    def __init__(self, config_model, config_data):
        ModuleBase.__init__(self)
        self.config_model = config_model
        self.config_data = config_data

        with open(config_data.vocab_file, "rb") as f:
            id2w = pickle.load(f)
        self.id2w = id2w
        self.vocab_size = len(id2w)
        self.pad_token_id, self.bos_token_id = (0, 1)
        self.eos_token_id, self.unk_token_id = (2, 3)

        self.word_embedder = tx.modules.WordEmbedder(
            vocab_size=self.vocab_size, hparams=config_model.emb
        )
        self.pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=config_data.max_decoding_length,
            hparams=config_model.position_embedder_hparams,
        )

        self.encoder = TransformerEncoder(hparams=config_model.encoder)
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            output_layer=self.word_embedder.embedding,
            hparams=config_model.decoder,
        )

        self.smoothed_loss_func = LabelSmoothingLoss(
            label_confidence=self.config_model.loss_label_confidence,
            tgt_vocab_size=self.vocab_size,
            ignore_index=0,
        )

        self.step_iteration = 0

    def forward(  # type: ignore
        self,
        encoder_input,
        is_train_mode: Optional[bool],
        decoder_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        beam_width: Optional[int] = None,
    ):

        batch_size = encoder_input.size()[0]
        # (text sequence length excluding padding)
        encoder_input_length = (encoder_input != 0).int().sum(dim=1)

        if is_train_mode:
            self.train()

        else:
            self.eval()

        # Source word embedding
        src_word_embeds = self.word_embedder(
            encoder_input
        )
        src_word_embeds = (
            src_word_embeds * self.config_model.hidden_dim ** 0.5
        )

        # Position embedding (shared b/w source and target)
        src_seq_len = torch.full(
            (batch_size,), encoder_input.size()[1], dtype=torch.int32
        )
        src_seq_len = src_seq_len.to(device=encoder_input.device)

        src_pos_embeds = self.pos_embedder(
            sequence_length=src_seq_len
        )
        src_input_embedding = src_word_embeds + src_pos_embeds

        encoder_output = self.encoder(
            inputs=src_input_embedding, sequence_length=encoder_input_length
        )

        if is_train_mode:
            assert decoder_input is not None
            assert labels is not None

            tgt_word_embeds = self.word_embedder(
                decoder_input
            )
            tgt_word_embeds = (
                tgt_word_embeds * self.config_model.hidden_dim ** 0.5
            )
            tgt_seq_len = torch.full(
                (batch_size,), decoder_input.size()[1], dtype=torch.int32
            )
            tgt_seq_len = tgt_seq_len.to(device=decoder_input.device)

            tgt_pos_embeds = self.pos_embedder(
                sequence_length=tgt_seq_len
            )

            tgt_input_embedding = tgt_word_embeds + tgt_pos_embeds

            # For training
            outputs = self.decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                inputs=tgt_input_embedding,
                decoding_strategy="train_greedy",
            )
            labels = labels.to(device=outputs.logits.device)
            label_lengths = (labels != 0).long().sum(dim=1)
            label_lengths = label_lengths.to(device=outputs.logits.device)
            is_target = (labels != 0).float()
            mle_loss = self.smoothed_loss_func(
                outputs.logits, labels, label_lengths
            )
            mle_loss = (mle_loss * is_target).sum() / is_target.sum()
            return mle_loss
        else:
            start_tokens = encoder_input.new_full(
                (batch_size,), self.bos_token_id, dtype=torch.int32)

            if torch.cuda.is_available():
                start_tokens = start_tokens.cuda()

            def _embedding_fn(x, y):
                word_embed = self.word_embedder(x)
                scale = self.config_model.hidden_dim ** 0.5
                pos_embed = self.pos_embedder(y)
                return word_embed * scale + pos_embed

            predictions = self.decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                beam_width=beam_width,
                length_penalty=self.config_model.length_penalty,
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                embedding=_embedding_fn,
                max_decoding_length=self.config_data.max_decoding_length,
                decoding_strategy="infer_greedy"
            )
            # Uses the best sample by beam search
            return predictions


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        """
        :param label_confidence: the confidence weight on the ground truth label
        :param tgt_vocab_size: the size of the final classification
        :param ignore_index: The index in the vocabulary to ignore weight
        """
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size
        super(LabelSmoothingLoss, self).__init__()

        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(  # type: ignore
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        label_lengths: torch.LongTensor,
    ) -> torch.Tensor:
        """
        output (FloatTensor): batch_size x seq_length * n_classes
        target (LongTensor): batch_size * seq_length, specify the label target
        label_lengths(torch.LongTensor): specify the length of the labels
        """
        ori_shapes = (output.size(), target.size())
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        output = output.view(ori_shapes[0])
        model_prob = model_prob.view(ori_shapes[0])

        return sequence_softmax_cross_entropy(
            labels=model_prob, logits=output, sequence_length=label_lengths,
            average_across_batch=False, sum_over_timesteps=False,
        )
