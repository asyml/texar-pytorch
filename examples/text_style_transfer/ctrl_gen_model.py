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
"""Text style transfer
"""

# pylint: disable=invalid-name, too-many-locals

import torch
import torch.nn.functional as F

import texar as tx
from texar.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier
from texar.module_base import ModuleBase
from texar.utils import get_batch_size


class CtrlGenModel(ModuleBase):
    """Control
    """

    def __init__(self, vocab, hparams=None, device=None):

        super().__init__()

        self._hparams = tx.HParams(hparams, None)
        self.vocab = vocab
        self.embedder = WordEmbedder(
            vocab_size=vocab.size,
            hparams=self._hparams.embedder)

        self.encoder = UnidirectionalRNNEncoder(
            input_size=self._hparams.embedder.dim,
            hparams=self._hparams.encoder)

        self.label_connector = MLPTransformConnector(
            self._hparams.dim_c, linear_layer_dim=1)

        encoder_kwargs = self._hparams.encoder.rnn_cell.kwargs
        self.decoder = AttentionRNNDecoder(
            input_size=self._hparams.embedder.dim,
            encoder_output_size=encoder_kwargs.num_units,
            vocab_size=self.vocab.size,
            cell_input_fn=lambda inputs, attention: inputs,
            hparams=self._hparams.decoder)

        self.classifier = Conv1DClassifier(
            in_channels=self._hparams.embedder.dim,
            hparams=self._hparams.classifier)

        self.clas_embedder = WordEmbedder(
            vocab_size=self.vocab.size,
            hparams=self._hparams.embedder)

        decoder_kwargs = self._hparams.decoder.rnn_cell.kwargs
        self.connector = MLPTransformConnector(
            self.decoder.state_size,
            linear_layer_dim=decoder_kwargs.num_units)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.device = device

    def forward(self, inputs, gamma, lambda_g, mode='train'):
        """Builds the model.
        """

        # text_ids for encoder, with BOS token removed
        enc_text_ids = inputs['text_ids'][:, 1:].to(self.device)
        sequence_length = (inputs['length'] - 1).to(self.device)
        enc_outputs, final_state = self.encoder(self.embedder(enc_text_ids),
                                           sequence_length=sequence_length)
        z = final_state[:, self._hparams.dim_c:]

        # Encodes label

        # Gets the sentence representation: h = (c, z)
        labels = inputs['labels'].view(-1, 1).type(torch.FloatTensor)
        c = self.label_connector(labels)
        c_ = self.label_connector(1 - labels)
        h = torch.cat((c, z), 1)
        h_ = torch.cat((c_, z), 1)

        # Teacher-force decoding and the auto-encoding loss for G

        state_d = self.connector(h)

        helper = self.decoder.create_helper(
            embedding=self.embedder)
        g_outputs, _, _ = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=sequence_length,
            inputs=inputs['text_ids'],
            sequence_length=sequence_length,
            initial_state=state_d,
            helper=helper)

        loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=inputs['text_ids'][:, 1:],
            logits=g_outputs.logits,
            sequence_length=sequence_length,
            average_across_timesteps=True,
            sum_over_timesteps=False)

        # Gumbel-softmax decoding, used in training
        start_tokens = torch.ones_like(
            inputs['labels']) * self.vocab.bos_token_id
        end_token = self.vocab.eos_token_id
        start_tokens = start_tokens.to(dtype=torch.long, device=self.device)
        end_token = int(end_token)

        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            self.embedder.embedding, start_tokens, end_token, gamma)

        state_g = self.connector(h_)

        soft_outputs_, _, soft_length_, = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=sequence_length,
            helper=gumbel_helper,
            initial_state=state_g)

        # Greedy decoding, used in eval

        greedy_helper = self.decoder.create_helper(
            decoding_strategy='infer_greedy',
            start_tokens=start_tokens,
            end_token=end_token,
            embedding=self.embedder)
        outputs_, _, length_ = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=sequence_length,
            helper=greedy_helper,
            initial_state=state_g)

        # Classification loss for the classifier
        clas_inputs = self.clas_embedder(ids=inputs['text_ids'][:, 1:])
        clas_logits, clas_preds = self.classifier(
            input=clas_inputs.transpose(1, 2),
            sequence_length=sequence_length)
        loss_d_clas = self.loss_fn(
            clas_logits,
            inputs['labels'].to(dtype=torch.float, device=self.device))
        accu_d = tx.evals.accuracy(labels=inputs['labels'], preds=clas_preds)

        # Classification loss for the generator, based on soft samples
        g_clas_input = self.clas_embedder(
            soft_ids=soft_outputs_.sample_id)
        soft_logits, soft_preds = self.classifier(
            input=g_clas_input.transpose(1, 2).to(self.device),
            sequence_length=soft_length_)
        loss_g_clas = self.loss_fn(
            soft_logits,
            (1 - inputs['labels']).to(dtype=torch.float, device=self.device))

        # Accuracy on soft samples, for training progress monitoring
        accu_g = tx.evals.accuracy(labels=1 - inputs['labels'], preds=soft_preds)

        # Accuracy on greedy-decoded samples, for training progress monitoring
        max_dim = max(self._hparams.classifier.kernel_size)
        input_gdy = self.clas_embedder(ids=outputs_.sample_id).transpose(1, 2)
        if max_dim > input_gdy.size()[-1]:
            pad = (0, max_dim - input_gdy.size()[-1])
            input_gdy = F.pad(input_gdy, pad, 'constant', 0)

        _, gdy_preds = self.classifier(
            input=input_gdy,
            sequence_length=length_)
        accu_g_gdy = tx.evals.accuracy(
            labels=1 - inputs['labels'], preds=gdy_preds)

        # Aggregates losses
        loss_g = loss_g_ae + lambda_g * loss_g_clas
        loss_d = loss_d_clas

        # Interface tensors

        fetches_train_d = {
            "loss_d": loss_d,
            "accu_d": accu_d
        }

        fetches_train_g = {
            "loss_g": loss_g,
            "loss_g_ae": loss_g_ae,
            "loss_g_clas": loss_g_clas,
            "accu_g": accu_g,
            "accu_g_gdy": accu_g_gdy
        }

        fetches_eval = {}
        if mode != "train":
            losses = {
                "loss_g": loss_g,
                "loss_g_ae": loss_g_ae,
                "loss_g_clas": loss_g_clas,
                "loss_d": loss_d
            }
            metrics = {
                "accu_d": accu_d,
                "accu_g": accu_g,
                "accu_g_gdy": accu_g_gdy
            }
            samples = {
                "original": inputs['text_ids'][:, 1:],
                "transferred": outputs_.sample_id
            }
            fetches_eval = {"batch_size": get_batch_size(inputs['text_ids'])}
            fetches_eval.update(losses)
            fetches_eval.update(metrics)
            fetches_eval["samples"] = samples

        return fetches_train_d, fetches_train_g, fetches_eval
