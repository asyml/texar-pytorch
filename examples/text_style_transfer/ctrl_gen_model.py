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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

#import tensorflow as tf
import torch
from torch import nn

import texar as tx
from texar.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier
from texar.module_base import ModuleBase
from texar.core import get_train_op, AttentionWrapperState
from texar.utils import get_batch_size


class CtrlGenModel(ModuleBase):
    """Control
    """

    #def __init__(self, inputs, vocab, gamma, lambda_g, hparams=None):
    def __init__(self, vocab, hparams=None):

        super().__init__()

        self._hparams = tx.HParams(hparams, None)
        #self._build_model(inputs, vocab, gamma, lambda_g)
        self.vocab = vocab
        self.embedder = WordEmbedder(
            vocab_size=vocab.size,
            hparams=self._hparams.embedder)

        self.encoder = UnidirectionalRNNEncoder(input_size=100, hparams=self._hparams.encoder)
        self.label_connector = MLPTransformConnector(self._hparams.dim_c, linear_layer_dim=1)
        self.decoder = AttentionRNNDecoder(
            input_size=100,
            encoder_output_size=700,
            vocab_size=self.vocab.size,
            cell_input_fn=lambda inputs, attention: inputs,
            hparams=self._hparams.decoder)
        self.classifier = Conv1DClassifier(
            in_channels=100,
            hparams=self._hparams.classifier)
        self.clas_embedder = WordEmbedder(
            vocab_size=self.vocab.size,
            hparams=self._hparams.embedder)
        print("self.clas_embedder", self.clas_embedder.dim)
        self.loss_d = torch.nn.BCEWithLogitsLoss()
        self.loss_g = torch.nn.BCEWithLogitsLoss()

    #def _build_model(self, inputs, vocab, gamma, lambda_g):
    def forward(self, inputs, gamma, lambda_g):
        """Builds the model.
        """

        # text_ids for encoder, with BOS token removed
        enc_text_ids = inputs['text_ids'][:, 1:]

        enc_outputs, final_state = self.encoder(self.embedder(enc_text_ids),
                                           sequence_length=inputs['length']-1)
        z = final_state[:, self._hparams.dim_c:]

        # Encodes label
        #label_connector = MLPTransformConnector(self._hparams.dim_c)

        # Gets the sentence representation: h = (c, z)
        #labels = tf.to_float(tf.reshape(inputs['labels'], [-1, 1]))
        labels = inputs['labels'].view(-1, 1).type(torch.FloatTensor)
        c = self.label_connector(labels)
        c_ = self.label_connector(1 - labels)
        #h = tf.concat([c, z], 1)
        h = torch.cat((c, z), 1)
        #h_ = tf.concat([c_, z], 1)
        h_ = torch.cat((c_, z), 1)

        # Teacher-force decoding and the auto-encoding loss for G

        connector = MLPTransformConnector(self.decoder.state_size, linear_layer_dim=h.size(-1))

        state = self.decoder._cell.zero_state(64)
        state = state._replace(cell_state=connector(h))

        helper = self.decoder.create_helper(
            embedding=self.embedder)
        g_outputs, _, _ = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length']-1,
            inputs=inputs['text_ids'],
            sequence_length=inputs['length']-1,
            initial_state=state,
            helper=helper)

        loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=inputs['text_ids'][:, 1:],
            logits=g_outputs.logits,
            sequence_length=inputs['length']-1,
            average_across_timesteps=True,
            sum_over_timesteps=False)

        # Gumbel-softmax decoding, used in training
        start_tokens = torch.ones_like(inputs['labels']) * self.vocab.bos_token_id
        end_token = self.vocab.eos_token_id
        start_tokens = start_tokens.type(torch.long)
        end_token = int(end_token)

        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            self.embedder.embedding, start_tokens, end_token, gamma)

        state_ = self.decoder._cell.zero_state(64)
        state_ = state_._replace(cell_state=connector(h_))
        soft_outputs_, _, soft_length_, = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length']-1,
            helper=gumbel_helper,
            initial_state=state_)

        # Greedy decoding, used in eval

        greedy_helper = self.decoder.create_helper(
            decoding_strategy='infer_greedy',
            start_tokens=start_tokens,
            end_token=end_token,
            embedding=self.embedder)
        outputs_, _, length_ = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length']-1,
            helper=greedy_helper,
            initial_state=state_)

        # Classification loss for the classifier
        id_inputs = inputs['text_ids'][:, 1:]

        clas_logits, clas_preds = self.classifier(
            input=self.clas_embedder(ids=inputs['text_ids'][:, 1:]).transpose(1, 2),
            sequence_length=inputs['length']-1)

        loss_d_clas = self.loss_d(clas_logits, inputs['labels'].type(torch.FloatTensor))
        #loss_d_clas = tf.reduce_mean(loss_d_clas)
        loss_d_clas = loss_d_clas.mean()
        accu_d = tx.evals.accuracy(labels=inputs['labels'], preds=clas_preds)

        # Classification loss for the generator, based on soft samples
        soft_logits, soft_preds = self.classifier(
            input=self.clas_embedder(soft_ids=soft_outputs_.sample_id).transpose(1, 2),
            sequence_length=soft_length_)

        loss_g_clas = self.loss_g(soft_logits, (1 - inputs['labels']).type(torch.FloatTensor))
        #loss_g_clas = tf.reduce_mean(loss_g_clas)
        loss_g_clas = loss_g_clas.mean()
        # Accuracy on soft samples, for training progress monitoring
        accu_g = tx.evals.accuracy(labels=1-inputs['labels'], preds=soft_preds)

        # Accuracy on greedy-decoded samples, for training progress monitoring
        #print("outputs_.sample_id: {}".format(outputs_.sample_id.size()))
        '''_, gdy_preds = self.classifier(
            input=self.clas_embedder(ids=outputs_.sample_id).transpose(1, 2),
            sequence_length=length_)
        accu_g_gdy = tx.evals.accuracy(
            labels=1-inputs['labels'], preds=gdy_preds)'''

        # Aggregates losses
        loss_g = loss_g_ae + lambda_g * loss_g_clas
        loss_d = loss_d_clas

        # Interface tensors
        losses = {
            "loss_g": loss_g,
            "loss_g_ae": loss_g_ae,
            "loss_g_clas": loss_g_clas,
            "loss_d": loss_d_clas
        }
        metrics = {
            "accu_d": accu_d,
            "accu_g": accu_g,
            #"accu_g_gdy": accu_g_gdy,
        }
        samples = {
            "original": inputs['text_ids'][:, 1:],
            "transferred": outputs_.sample_id
        }

        fetches_train_g = {
            "loss_g": loss_g,
            "loss_g_ae": losses["loss_g_ae"],
            "loss_g_clas": losses["loss_g_clas"],
            "accu_g": metrics["accu_g"],
            #"accu_g_gdy": metrics["accu_g_gdy"],
        }

        fetches_train_d = {
            "loss_d": loss_d_clas,
            "accu_d": metrics["accu_d"]
        }
        fetches_eval = {"batch_size": get_batch_size(inputs['text_ids'])}
        fetches_eval.update(losses)
        fetches_eval.update(metrics)
        fetches_eval.update(samples)

        return fetches_train_d, fetches_train_g, fetches_eval



