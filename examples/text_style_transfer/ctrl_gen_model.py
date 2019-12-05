# Copyright 2018 The Texar Authors. All Rights Reserved.
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
"""Text style transfer
"""

# pylint: disable=invalid-name, too-many-locals

import torch
import torch.nn as nn

import texar.torch as tx
from texar.torch.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier
from texar.torch.utils import get_batch_size, collect_trainable_variables


class CtrlGenModel(nn.Module):
    """Control
    """
    def __init__(self, vocab: tx.data.Vocab, hparams=None):
        super().__init__()
        self.vocab = vocab

        self._hparams = tx.HParams(hparams, None)

        self.embedder = WordEmbedder(vocab_size=self.vocab.size,
                                     hparams=self._hparams.embedder)

        self.encoder = UnidirectionalRNNEncoder(
            input_size=self.embedder.dim,
            hparams=self._hparams.encoder)  # type: UnidirectionalRNNEncoder

        # Encodes label
        self.label_connector = MLPTransformConnector(
            output_size=self._hparams.dim_c,
            linear_layer_dim=1)

        # Teacher-force decoding and the auto-encoding loss for G
        self.decoder = AttentionRNNDecoder(
            input_size=self.embedder.dim,
            encoder_output_size=(self.encoder.cell.hidden_size),
            vocab_size=self.vocab.size,
            token_embedder=self.embedder,
            hparams=self._hparams.decoder)

        self.connector = MLPTransformConnector(
            output_size=self.decoder.output_size,
            linear_layer_dim=(self._hparams.dim_c + self._hparams.dim_z))

        self.classifier = Conv1DClassifier(
            in_channels=self.embedder.dim,
            in_features=self._hparams.max_seq_length,
            hparams=self._hparams.classifier)

        self.class_embedder = WordEmbedder(vocab_size=self.vocab.size,
                                          hparams=self._hparams.embedder)

        # Creates optimizers
        self.g_vars = collect_trainable_variables(
            [self.embedder, self.encoder, self.label_connector,
             self.connector, self.decoder])

        self.d_vars = collect_trainable_variables(
            [self.class_embedder, self.classifier])

    def forward_D(self, inputs, f_labels):

        # Classification loss for the classifier
        # Get inputs in correct format, [batch_size, channels, seq_length]
        class_inputs = self.class_embedder(ids=inputs['text_ids'][:, 1:])
        class_logits, class_preds = self.classifier(
            input=class_inputs,
            sequence_length=inputs['length'] - 1)

        sig_ce_logits_loss = nn.BCEWithLogitsLoss()

        loss_d = sig_ce_logits_loss(class_logits, f_labels)
        accu_d = tx.evals.accuracy(labels=f_labels,
                                   preds=class_preds)
        return {
            "loss_d": loss_d,
            "accu_d": accu_d
        }

    def forward_G(self, inputs, f_labels, gamma, lambda_g, mode):

        # text_ids for encoder, with BOS token removed
        enc_text_ids = inputs['text_ids'][:, 1:].long()
        enc_inputs = self.embedder(enc_text_ids)
        enc_outputs, final_state = self.encoder(
            enc_inputs,
            sequence_length=inputs['length'] - 1)
        z = final_state[:, self._hparams.dim_c:]

        labels = inputs['labels'].view(-1, 1).float()

        c = self.label_connector(labels)
        c_ = self.label_connector(1 - labels)
        h = torch.cat([c, z], dim=1)
        h_ = torch.cat([c_, z], dim=1)

        # Gumbel-softmax decoding, used in training
        start_tokens = torch.ones_like(inputs['labels'].long()) * \
                       self.vocab.bos_token_id
        end_token = self.vocab.eos_token_id

        if mode == 'train':
            g_outputs, _, _ = self.decoder(
                memory=enc_outputs,
                memory_sequence_length=inputs['length'] - 1,
                initial_state=self.connector(h),
                inputs=inputs['text_ids'],
                embedding=self.embedder,
                sequence_length=inputs['length'] - 1
            )

            loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=inputs['text_ids'][:, 1:],
                logits=g_outputs.logits,
                sequence_length=inputs['length'] - 1,
                average_across_timesteps=True,
                sum_over_timesteps=False
            )

        else:
            # for eval, there is no loss
            loss_g_ae = 0

        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            start_tokens=start_tokens,
            end_token=end_token,
            tau=gamma)

        soft_outputs_, _, soft_length_, = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length'] - 1,
            helper=gumbel_helper,
            initial_state=self.connector(h_))

        # Greedy decoding, used in eval
        outputs_, _, length_ = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length'] - 1,
            decoding_strategy='infer_greedy',
            initial_state=self.connector(h_),
            embedding=self.embedder,
            start_tokens=start_tokens,
            end_token=end_token)

        # Get inputs in correct format, [batch_size, channels, seq_length]
        soft_inputs = self.class_embedder(soft_ids=soft_outputs_.sample_id)
        soft_logits, soft_preds = self.classifier(
            input=soft_inputs,
            sequence_length=soft_length_)

        sig_ce_logits_loss = nn.BCEWithLogitsLoss()

        loss_g_class = sig_ce_logits_loss(soft_logits, (1 - f_labels))

        # Accuracy on greedy-decoded samples, for training progress monitoring
        greedy_inputs = self.class_embedder(ids=outputs_.sample_id)
        _, gdy_preds = self.classifier(
            input=greedy_inputs,
            sequence_length=length_)

        accu_g_gdy = tx.evals.accuracy(
            labels=1 - f_labels, preds=gdy_preds)

        # Accuracy on soft samples, for training progress monitoring
        accu_g = tx.evals.accuracy(labels=1 - f_labels,
                                   preds=soft_preds)
        loss_g = loss_g_ae + lambda_g * loss_g_class
        ret = {
            "loss_g": loss_g,
            "loss_g_ae": loss_g_ae,
            "loss_g_class": loss_g_class,
            "accu_g": accu_g,
            "accu_g_gdy": accu_g_gdy,
        }
        if mode == 'eval':
            ret.update({'outputs': outputs_})
        return ret

    def forward(self, inputs, gamma, lambda_g, mode, component=None):

        f_labels = inputs['labels'].float()
        if mode == 'train':
            if component == 'D':
                ret_d = self.forward_D(inputs, f_labels)
                return ret_d

            elif component == 'G':
                ret_g = self.forward_G(inputs, f_labels, gamma, lambda_g, mode)
                return ret_g

        else:
            ret_d = self.forward_D(inputs, f_labels)
            ret_g = self.forward_G(inputs, f_labels, gamma, lambda_g, mode)
            rets = {
                "batch_size": get_batch_size(inputs['text_ids']),
                "loss_g": ret_g['loss_g'],
                "loss_g_ae": ret_g['loss_g_ae'],
                "loss_g_class": ret_g['loss_g_class'],
                "loss_d": ret_d['loss_d'],
                "accu_d": ret_d['accu_d'],
                "accu_g": ret_g['accu_g'],
                "accu_g_gdy": ret_g['accu_g_gdy']
            }
            samples = {
                "original": inputs['text_ids'][:, 1:],
                "transferred": ret_g['outputs'].sample_id
            }
            return rets, samples
