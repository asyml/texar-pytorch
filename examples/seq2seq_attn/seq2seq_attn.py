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
"""Attentional Seq2seq.
"""

import argparse
import importlib
from typing import Any

import torch
import torch.nn as nn

import texar.torch as tx

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config-model', type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    '--config-data', type=str, default="config_iwslt14",
    help="The dataset config.")
args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2SeqAttn(nn.Module):

    def __init__(self, train_data):
        super().__init__()

        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id

        self.source_embedder = tx.modules.WordEmbedder(
            vocab_size=self.source_vocab_size,
            hparams=config_model.embedder)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=config_model.embedder)

        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=self.source_embedder.dim,
            hparams=config_model.encoder)

        self.decoder = tx.modules.AttentionRNNDecoder(
            token_embedder=self.target_embedder,
            encoder_output_size=(self.encoder.cell_fw.hidden_size +
                                 self.encoder.cell_bw.hidden_size),
            input_size=self.target_embedder.dim,
            vocab_size=self.target_vocab_size,
            hparams=config_model.decoder)

    def forward(self, batch, mode):
        enc_outputs, _ = self.encoder(
            inputs=self.source_embedder(batch['source_text_ids']),
            sequence_length=batch['source_length'])

        memory = torch.cat(enc_outputs, dim=2)

        if mode == "train":
            helper_train = self.decoder.create_helper(
                decoding_strategy="train_greedy")

            training_outputs, _, _ = self.decoder(
                memory=memory,
                memory_sequence_length=batch['source_length'],
                helper=helper_train,
                inputs=batch['target_text_ids'][:, :-1],
                sequence_length=batch['target_length'] - 1)

            mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['target_text_ids'][:, 1:],
                logits=training_outputs.logits,
                sequence_length=batch['target_length'] - 1)

            return mle_loss
        else:
            start_tokens = memory.new_full(
                batch['target_length'].size(), self.bos_token_id,
                dtype=torch.int64)

            infer_outputs = self.decoder(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                memory=memory,
                memory_sequence_length=batch['source_length'],
                beam_width=config_model.beam_width)

            return infer_outputs


def main() -> None:
    train_data = tx.data.PairedTextData(
        hparams=config_data.train, device=device)
    val_data = tx.data.PairedTextData(
        hparams=config_data.val, device=device)
    test_data = tx.data.PairedTextData(
        hparams=config_data.test, device=device)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    model = Seq2SeqAttn(train_data)
    model.to(device)
    train_op = tx.core.get_train_op(
        params=model.parameters(), hparams=config_model.opt)

    def _train_epoch():
        data_iterator.switch_to_train_data()
        model.train()

        step = 0
        for batch in data_iterator:
            loss = model(batch, mode="train")
            loss.backward()
            train_op()
            if step % config_data.display == 0:
                print("step={}, loss={:.4f}".format(step, loss))
            step += 1

    @torch.no_grad()
    def _eval_epoch(mode):
        if mode == 'val':
            data_iterator.switch_to_val_data()
        else:
            data_iterator.switch_to_test_data()
        model.eval()

        refs, hypos = [], []
        for batch in data_iterator:
            infer_outputs = model(batch, mode="val")
            output_ids = infer_outputs["sample_id"][:, :, 0].cpu()
            target_texts_ori = [text[1:] for text in batch['target_text']]
            target_texts = tx.utils.strip_special_tokens(
                target_texts_ori, is_token_list=True)
            output_texts = tx.data.vocabulary.map_ids_to_strs(
                ids=output_ids, vocab=val_data.target_vocab)

            for hypo, ref in zip(output_texts, target_texts):
                hypos.append(hypo)
                refs.append([ref])

        return tx.evals.corpus_bleu_moses(
            list_of_references=refs, hypotheses=hypos)

    best_val_bleu = -1.
    for i in range(config_data.num_epochs):
        _train_epoch()

        val_bleu = _eval_epoch('val')
        best_val_bleu = max(best_val_bleu, val_bleu)
        print('val epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(
            i, val_bleu, best_val_bleu))

        test_bleu = _eval_epoch('test')
        print('test epoch={}, BLEU={:.4f}'.format(i, test_bleu))

        print('=' * 50)


if __name__ == '__main__':
    main()
