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
"""Example for building the language model.
"""

from typing import Any, Dict, List, Tuple

import argparse
import importlib
import time

import torch
import torch.nn as nn

import texar.torch as tx

from ptb_reader import prepare_data, ptb_iterator

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path', type=str, default='./',
    help="Directory containing PTB raw data (e.g., ptb.train.txt). "
         "E.g., ./simple-examples/data. If not exists, "
         "the directory will be created and PTB raw data will be downloaded.")
parser.add_argument(
    '--config', type=str, default='config_small',
    help='The config to use.')
args = parser.parse_args()

config: Any = importlib.import_module(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PTBLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, hparams: Dict[str, Any]):
        super().__init__()

        self.embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams=hparams['embedder'])
        self.decoder = tx.modules.BasicRNNDecoder(
            token_embedder=self.embedder,
            input_size=config.hidden_size, vocab_size=vocab_size,
            hparams=hparams['decoder'])

    def forward(self,  # type: ignore
                inputs: torch.Tensor, targets: torch.Tensor,
                state: List[Tuple[torch.Tensor]]):
        outputs, final_state, seq_lengths = self.decoder(
            decoding_strategy="train_greedy",
            impute_finished=False,
            inputs=inputs,
            sequence_length=torch.tensor(
                [config.num_steps] * config.batch_size),
            initial_state=state)

        mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=targets,
            logits=outputs.logits,
            sequence_length=seq_lengths)

        return mle_loss, final_state


def main() -> None:
    # Data
    batch_size = config.batch_size
    num_steps = config.num_steps
    num_epochs = config.num_epochs

    data = prepare_data(args.data_path)

    hparams = {
        'embedder': config.emb,
        'decoder': {"rnn_cell": config.cell},
    }
    model = PTBLanguageModel(vocab_size=data["vocab_size"], hparams=hparams)
    model.to(device)
    train_op = tx.core.get_train_op(params=model.parameters(),
                                    hparams=config.opt)

    def _run_epoch(data_iter, is_train=False, verbose=False):
        start_time = time.time()
        loss = 0.
        iters = 0
        state = None

        if is_train:
            model.train()
            epoch_size = ((len(data["train_text_id"]) // batch_size - 1) //
                          num_steps)
        else:
            model.eval()

        for step, (x, y) in enumerate(data_iter):
            loss_, state_ = model(inputs=torch.tensor(x),
                                  targets=torch.tensor(y), state=state)
            if is_train:
                loss_.backward()
                train_op()
            loss += loss_
            state = [(state_[0][0].detach(), state_[0][1].detach()),
                     (state_[1][0].detach(), state_[1][1].detach())]
            iters += num_steps

            ppl = torch.exp(loss / iters)
            if verbose and is_train and step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      ((step + 1) * 1.0 / epoch_size, ppl,
                       iters * batch_size / (time.time() - start_time)))

        ppl = torch.exp(loss / iters)
        return ppl

    for epoch in range(num_epochs):
        # Train
        train_data_iter = ptb_iterator(
            data["train_text_id"], batch_size, num_steps)
        train_ppl = _run_epoch(train_data_iter, is_train=True, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (epoch, train_ppl))
        train_op(use_scheduler=True)  # type: ignore

        # Valid
        valid_data_iter = ptb_iterator(
            data["valid_text_id"], batch_size, num_steps)
        valid_ppl = _run_epoch(valid_data_iter)
        print("Epoch: %d Valid Perplexity: %.3f" % (epoch, valid_ppl))

    # Test
    test_data_iter = ptb_iterator(
        data["test_text_id"], batch_size, num_steps)
    test_ppl = _run_epoch(test_data_iter)
    print("Test Perplexity: %.3f" % test_ppl)


if __name__ == '__main__':
    main()
