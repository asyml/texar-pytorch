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
"""Example for building a sentence convolutional classifier.

Use `./sst_data_preprocessor.py` to download and clean the SST binary data.

To run:

$ python clas_main.py --config=config_kim
"""

from typing import Any, Dict, Tuple

import argparse
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import texar.torch as tx

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str, default='config_kim',
    help='The config to use.')
args = parser.parse_args()

config: Any = importlib.import_module(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentenceClassifier(nn.Module):

    def __init__(self, vocab_size: int, max_seq_length: int,
                 emb_dim: int, hparams: Dict[str, Any]):
        super().__init__()

        self.embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams=hparams['embedder'])
        self.classifier = tx.modules.Conv1DClassifier(
            in_channels=max_seq_length,
            in_features=emb_dim, hparams=hparams['classifier'])

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, pred = self.classifier(
            self.embedder(batch['sentence_text_ids']))
        loss = F.cross_entropy(logits, batch['label'])
        return pred, loss


def main() -> None:
    # Data
    train_data = tx.data.MultiAlignedData(config.train_data, device=device)
    val_data = tx.data.MultiAlignedData(config.val_data, device=device)
    test_data = tx.data.MultiAlignedData(config.test_data, device=device)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    hparams = {
        'embedder': config.emb,
        'classifier': config.clas
    }
    model = SentenceClassifier(vocab_size=train_data.vocab('sentence').size,
                               max_seq_length=config.max_seq_length,
                               emb_dim=config.emb_dim,
                               hparams=hparams)
    model.to(device)
    train_op = tx.core.get_train_op(params=model.parameters(),
                                    hparams=config.opt)

    def _run_epoch(mode, epoch):

        step = 0
        avg_rec = tx.utils.AverageRecorder()
        for batch in data_iterator:
            pred, loss = model(batch)
            if mode == "train":
                loss.backward()
                train_op()
            accu = tx.evals.accuracy(batch['label'], pred)
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"epoch: {epoch:2} step: {step:4} accu: {accu:.4f}")

            batch_size = batch['label'].size(0)
            avg_rec.add([accu], batch_size)

        return avg_rec.avg(0)

    best_val_accu = -1
    for epoch in range(config.num_epochs):
        # Train
        data_iterator.switch_to_train_data()
        model.train()
        train_accu = _run_epoch("train", epoch)

        # Val
        data_iterator.switch_to_val_data()
        model.eval()
        val_accu = _run_epoch("val", epoch)
        print(f'epoch: {epoch:2} train accu: {train_accu:.4f} '
              f'val accu: {val_accu:.4f}')

        # Test
        if val_accu > best_val_accu:
            best_val_accu = val_accu
            data_iterator.switch_to_test_data()
            model.eval()
            test_accu = _run_epoch("test", epoch)
            print(f'test accu: {test_accu:.4f}')


if __name__ == '__main__':
    main()
