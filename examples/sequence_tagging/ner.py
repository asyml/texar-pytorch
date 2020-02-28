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
Sequence tagging.
"""

from typing import Any

import argparse
import importlib
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import texar.torch as tx

from conll_reader import (create_vocabs, construct_init_word_vecs,
                          iterate_batch, load_glove, read_data, MAX_CHAR_LENGTH)
from conll_writer import CoNLLWriter
from scores import scores


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-path", default="./data",
    help="Directory containing NER data (e.g., eng.train.bio.conll).")
parser.add_argument(
    "--train", default="eng.train.bio.conll",
    help="The file name of the training data.")
parser.add_argument(
    "--dev", default="eng.dev.bio.conll",
    help="The file name of the dev data.")
parser.add_argument(
    "--test", default="eng.test.bio.conll",
    help="The file name of the testing data.")
parser.add_argument(
    "--embedding", default="glove.6B.100d.txt",
    help="The file name of the GloVe embedding.")
parser.add_argument(
    "--config", default="config", help="The configurations to use.")
args = parser.parse_args()

config: Any = importlib.import_module(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = os.path.join(args.data_path, args.train)
dev_path = os.path.join(args.data_path, args.dev)
test_path = os.path.join(args.data_path, args.test)
embedding_path = os.path.join(args.data_path, args.embedding)

EMBEDD_DIM = config.embed_dim
CHAR_DIM = config.char_dim

# Prepares/loads data
if config.load_glove:
    print('loading GloVe embedding...')
    glove_dict = load_glove(embedding_path, EMBEDD_DIM)
else:
    glove_dict = None

(word_vocab, char_vocab, ner_vocab), (i2w, i2n) = create_vocabs(
    train_path, dev_path, test_path, glove_dict=glove_dict)

data_train = read_data(train_path, word_vocab, char_vocab, ner_vocab)
data_dev = read_data(dev_path, word_vocab, char_vocab, ner_vocab)
data_test = read_data(test_path, word_vocab, char_vocab, ner_vocab)

scale = np.sqrt(3.0 / EMBEDD_DIM)
word_vecs = np.random.uniform(
    -scale, scale, [len(word_vocab), EMBEDD_DIM]).astype(np.float32)
if config.load_glove:
    word_vecs = construct_init_word_vecs(word_vocab, word_vecs, glove_dict)

scale = np.sqrt(3.0 / CHAR_DIM)
char_vecs = np.random.uniform(
    -scale, scale, [len(char_vocab), CHAR_DIM]).astype(np.float32)


class NER(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedder = tx.modules.WordEmbedder(
            vocab_size=len(word_vecs), init_value=torch.tensor(word_vecs),
            hparams=config.emb)
        self.char_embedder = tx.modules.WordEmbedder(
            vocab_size=len(char_vecs), init_value=torch.tensor(char_vecs),
            hparams=config.char_emb)
        self.char_encoder = tx.modules.Conv1DEncoder(
            in_channels=MAX_CHAR_LENGTH, in_features=CHAR_DIM,
            hparams=config.conv)
        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=(EMBEDD_DIM + CHAR_DIM),
            hparams={"rnn_cell_fw": config.cell, "rnn_cell_bw": config.cell})

        self.dropout_1 = nn.Dropout(p=0.33)
        self.dense_1 = nn.Linear(in_features=2 * config.hidden_size,
                                 out_features=config.tag_space)
        self.dropout_2 = nn.Dropout(p=(1 - config.keep_prob))
        self.dense_2 = nn.Linear(in_features=config.tag_space,
                                 out_features=len(ner_vocab))

    def forward(self, inputs, chars, targets, masks, seq_lengths, mode):
        emb_inputs = self.embedder(inputs)
        emb_chars = self.char_embedder(chars)
        char_shape = emb_chars.shape
        emb_chars = torch.reshape(emb_chars, (-1, char_shape[2], CHAR_DIM))

        char_outputs = self.char_encoder(emb_chars)
        char_outputs = torch.reshape(char_outputs, (
            char_shape[0], char_shape[1], CHAR_DIM))
        emb_inputs = torch.cat((emb_inputs, char_outputs), dim=2)

        emb_inputs = self.dropout_1(emb_inputs)
        outputs, _ = self.encoder(emb_inputs, sequence_length=seq_lengths)
        outputs = torch.cat(outputs, dim=2)
        rnn_shape = outputs.shape
        outputs = torch.reshape(outputs, (-1, 2 * config.hidden_size))
        outputs = F.elu(self.dense_1(outputs))
        outputs = self.dropout_2(outputs)
        logits = self.dense_2(outputs)
        logits = torch.reshape(
            logits, (rnn_shape[0], rnn_shape[1], len(ner_vocab)))
        predicts = torch.argmax(logits, dim=2)
        corrects = torch.sum(torch.eq(predicts, targets) * masks)

        if mode == 'train':
            mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=targets,
                logits=logits,
                sequence_length=seq_lengths,
                average_across_batch=True,
                average_across_timesteps=True,
                sum_over_timesteps=False)
            return mle_loss, corrects
        else:
            return predicts


def main() -> None:
    model = NER()
    model.to(device)
    train_op = tx.core.get_train_op(params=model.parameters(),
                                    hparams=config.opt)

    def _train_epoch(epoch_):
        model.train()

        start_time = time.time()
        loss = 0.
        corr = 0.
        num_tokens = 0.

        num_inst = 0
        for batch in iterate_batch(data_train, config.batch_size, shuffle=True):
            word, char, ner, mask, length = batch
            mle_loss, correct = model(torch.tensor(word, device=device),
                                      torch.tensor(char, device=device),
                                      torch.tensor(ner, device=device),
                                      torch.tensor(mask, device=device),
                                      torch.tensor(length, device=device),
                                      'train')
            mle_loss.backward()
            train_op()

            nums = np.sum(length)
            num_inst += len(word)
            loss += mle_loss * nums
            corr += correct
            num_tokens += nums

            print("train: %d (%d/%d) loss: %.4f, acc: %.2f%%" % (
                epoch_, num_inst, len(data_train), loss / num_tokens,
                corr / num_tokens * 100))

        print("train: %d loss: %.4f, acc: %.2f%%, time: %.2fs" % (
            epoch_, loss / num_tokens, corr / num_tokens * 100,
            time.time() - start_time))

    @torch.no_grad()
    def _eval_epoch(epoch_, mode):
        model.eval()

        file_name = 'tmp/%s%d' % (mode, epoch_)
        writer = CoNLLWriter(i2w, i2n)
        writer.start(file_name)
        data = data_dev if mode == 'dev' else data_test

        for batch in iterate_batch(data, config.batch_size, shuffle=False):
            word, char, ner, mask, length = batch
            predictions = model(torch.tensor(word, device=device),
                                torch.tensor(char, device=device),
                                torch.tensor(ner, device=device),
                                torch.tensor(mask, device=device),
                                torch.tensor(length, device=device),
                                mode)

            writer.write(word, predictions.numpy(), ner, length)
        writer.close()
        acc_, precision_, recall_, f1_ = scores(file_name)
        print(
            '%s acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (
                mode, acc_, precision_, recall_, f1_))
        return acc_, precision_, recall_, f1_

    dev_f1 = 0.0
    dev_acc = 0.0
    dev_precision = 0.0
    dev_recall = 0.0
    best_epoch = 0

    test_f1 = 0.0
    test_acc = 0.0
    test_prec = 0.0
    test_recall = 0.0

    tx.utils.maybe_create_dir('./tmp')

    for epoch in range(config.num_epochs):
        _train_epoch(epoch)
        acc, precision, recall, f1 = _eval_epoch(epoch, 'dev')
        if dev_f1 < f1:
            dev_f1 = f1
            dev_acc = acc
            dev_precision = precision
            dev_recall = recall
            best_epoch = epoch
            test_acc, test_prec, test_recall, test_f1 = \
                _eval_epoch(epoch, 'test')
        print('best acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, '
              'F1: %.2f%%, epoch: %d' % (dev_acc, dev_precision, dev_recall,
                                         dev_f1, best_epoch))
        print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, '
              'F1: %.2f%%, epoch: %d' % (test_acc, test_prec, test_recall,
                                         test_f1, best_epoch))
        print('---------------------------------------------------')


if __name__ == '__main__':
    main()
