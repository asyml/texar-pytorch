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
from torch import nn

import texar as tx
from texar.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier
from texar.module_base import ModuleBase
from texar.core import AttentionWrapperState
from texar.utils import get_batch_size


class CtrlGenModel(ModuleBase):
    """Control
    """

    def __init__(self, vocab, hparams=None):

        super().__init__()

        self._hparams = tx.HParams(hparams, None)
        self.vocab = vocab
        self.embedder = WordEmbedder(
            vocab_size=vocab.size,
            hparams=self._hparams.embedder).cuda()

        self.encoder = UnidirectionalRNNEncoder(
            input_size=100, hparams=self._hparams.encoder).cuda()

        self.label_connector = MLPTransformConnector(
            self._hparams.dim_c, linear_layer_dim=1).cuda()

        self.decoder = AttentionRNNDecoder(
            input_size=100,
            encoder_output_size=700,
            vocab_size=self.vocab.size,
            cell_input_fn=lambda inputs, attention: inputs,
            hparams=self._hparams.decoder).cuda()

        self.classifier = Conv1DClassifier(
            in_channels=100,
            hparams=self._hparams.classifier).cuda()

        self.clas_embedder = WordEmbedder(
            vocab_size=self.vocab.size,
            hparams=self._hparams.embedder).cuda()

        self.connector = MLPTransformConnector(
            self.decoder.state_size, linear_layer_dim=700).cuda()
        self.loss_fn_d = torch.nn.BCEWithLogitsLoss().to("cuda:0")
        self.loss_fn_g = torch.nn.BCEWithLogitsLoss().to("cuda:0")

    def forward(self, inputs, gamma, lambda_g, mode='train'):
        """Builds the model.
        """

        # text_ids for encoder, with BOS token removed
        enc_text_ids = inputs['text_ids'][:, 1:].cuda()
        enc_outputs, final_state = self.encoder(self.embedder(enc_text_ids),
                                           sequence_length=(inputs['length']-1).cuda())
        z = final_state[:, self._hparams.dim_c:].cuda()

        # Encodes label

        # Gets the sentence representation: h = (c, z)
        labels = inputs['labels'].view(-1, 1).type(torch.FloatTensor)
        c = self.label_connector(labels)
        c_ = self.label_connector(1 - labels)
        h = torch.cat((c, z), 1)
        h_ = torch.cat((c_, z), 1)

        # Teacher-force decoding and the auto-encoding loss for G

        state_d = self.decoder._cell.zero_state(64)
        state_d = state_d._replace(cell_state=self.connector(h))

        helper = self.decoder.create_helper(
            embedding=self.embedder)
        g_outputs, _, _ = self.decoder(
            memory=enc_outputs.cuda(),
            memory_sequence_length=(inputs['length']-1).cuda(),
            inputs=inputs['text_ids'].cuda(),
            sequence_length=(inputs['length']-1).cuda(),
            initial_state=state_d,
            helper=helper)

        loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=inputs['text_ids'][:, 1:].cuda(),
            logits=g_outputs.logits,
            sequence_length=(inputs['length']-1).cuda(),
            average_across_timesteps=True,
            sum_over_timesteps=False)

        # Gumbel-softmax decoding, used in training
        start_tokens = torch.ones_like(
            inputs['labels']) * self.vocab.bos_token_id
        end_token = self.vocab.eos_token_id
        start_tokens = start_tokens.type(torch.long).cuda()
        end_token = int(end_token)

        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            self.embedder.embedding, start_tokens, end_token, gamma)

        state_g = self.decoder._cell.zero_state(64)
        state_g = state_g._replace(cell_state=self.connector(h_))
        soft_outputs_, _, soft_length_, = self.decoder(
            memory=enc_outputs.cuda(),
            memory_sequence_length=(inputs['length']-1).cuda(),
            helper=gumbel_helper,
            initial_state=state_g)

        # Greedy decoding, used in eval

        greedy_helper = self.decoder.create_helper(
            decoding_strategy='infer_greedy',
            start_tokens=start_tokens.cuda(),
            end_token=end_token,
            embedding=self.embedder)
        outputs_, _, length_ = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length']-1,
            helper=greedy_helper,
            initial_state=state_g)

        # Classification loss for the classifier
        id_inputs = inputs['text_ids'][:, 1:]
        clas_logits, clas_preds = self.classifier(
            input=self.clas_embedder(ids=inputs['text_ids'][:, 1:]).transpose(1, 2),
            sequence_length=inputs['length']-1)
        loss_d_clas = self.loss_fn_d(clas_logits, inputs['labels'].type(torch.FloatTensor).cuda())
        accu_d = tx.evals.accuracy(labels=inputs['labels'], preds=clas_preds)

        # Classification loss for the generator, based on soft samples
        soft_logits, soft_preds = self.classifier(
            input=self.clas_embedder(soft_ids=soft_outputs_.sample_id).transpose(1, 2).cuda(),
            sequence_length=soft_length_)
        loss_g_clas = self.loss_fn_g(
            soft_logits,
            (1 - inputs['labels']).type(torch.FloatTensor).cuda())

        # Accuracy on soft samples, for training progress monitoring
        accu_g = tx.evals.accuracy(labels=1-inputs['labels'], preds=soft_preds)

        # Accuracy on greedy-decoded samples, for training progress monitoring
        '''print("self.clas_embedder(ids=outputs_.sample_id).transpose(1, 2)", self.clas_embedder(ids=outputs_.sample_id).transpose(1, 2).size(), length_.size())
        _, gdy_preds = self.classifier(
            input=self.clas_embedder(ids=outputs_.sample_id).transpose(1, 2),
            sequence_length=length_)
        accu_g_gdy = tx.evals.accuracy(
            labels=1-inputs['labels'], preds=gdy_preds)'''

        # Aggregates losses
        loss_g = loss_g_ae + lambda_g * loss_g_clas
        loss_d = loss_d_clas

        # Interface tensors

        fetches_train_d = {
            "loss_d": loss_d_clas,
            "accu_d": accu_d
        }

        fetches_train_g = {
            "loss_g": loss_g,
            "loss_g_ae": loss_g_ae,
            "loss_g_clas": loss_g_clas,
            "accu_g": accu_g
        }

        fetches_eval = {}
        if mode != "train":
            losses = {
                "loss_g": loss_g,
                "loss_g_ae": loss_g_ae,
                "loss_g_clas": loss_g_clas,
                "loss_d": loss_d_clas
            }
            metrics = {
                "accu_d": accu_d,
                "accu_g": accu_g,
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
