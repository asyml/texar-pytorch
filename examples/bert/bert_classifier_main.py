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
"""Example of building a sentence classifier based on pre-trained BERT
model.
"""

import argparse
import functools
import importlib
import logging
import os
import pprint

import torch

import texar as tx
import utils.model_utils as model_utils
from texar.models import BertClassifier
from texar.custom.optimizers import BertAdam
parser = argparse.ArgumentParser()

parser.add_argument(
    "--config_downstream", default="config_classifier",
    help="Configuration of the downstream part of the model")
parser.add_argument(
    "--config_format_bert", default="json",
    help="The configuration format. Set to 'json' if the BERT config file "
         "is in the same format of the official BERT config file. Set to "
         "'texar' if the BERT config file is in Texar format.")
parser.add_argument(
    "--config_bert_pretrain", default='uncased_L-12_H-768_A-12',
    help="The architecture of pre-trained BERT model to use.")
parser.add_argument(
    "--config_data", default="config_data", help="The dataset config.")
parser.add_argument(
    "--output_dir", default="output/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a model checkpoint (including bert modules) to restore from.")
parser.add_argument(
    "--do_train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do_eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do_test", action="store_true",
    help="Whether to run test on the test set.")

args = parser.parse_args()

config_data = importlib.import_module(args.config_data)

config_downstream = importlib.import_module(args.config_downstream)
config_downstream = {
    k: v for k, v in config_downstream.__dict__.items()
    if not k.startswith('__')}

logging.root.setLevel(logging.INFO)


def main():
    """
    Builds the model and runs.
    """

    tx.utils.maybe_create_dir(args.output_dir)

    # Loads data
    num_train_data = config_data.num_train_data

    # Builds BERT
    bert_pretrain_dir = f'bert_pretrained_models/{args.config_bert_pretrain}'
    if args.config_format_bert == "json":
        bert_config = model_utils.transform_bert_to_texar_config(
            os.path.join(bert_pretrain_dir, 'bert_config.json'))
    elif args.config_format_bert == 'texar':
        bert_config = importlib.import_module(
            f'bert_config_lib.config_model_{args.config_bert_pretrain}')
    else:
        raise ValueError('Unknown config_format_bert.')

    bert_hparams = BertClassifier.default_hparams()
    for key in bert_config.keys():
        bert_hparams[key] = bert_config[key]
    for key in config_downstream.keys():
        bert_hparams[key] = config_downstream[key]

    model = BertClassifier(hparams=bert_hparams)
    init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
    model_utils.init_bert_checkpoint(model, init_checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"Pretrained model loaded from {init_checkpoint}")

    # Builds learning rate decay scheduler
    static_lr = 2e-5

    num_train_steps = int(num_train_data / config_data.train_batch_size *
                          config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)

    vars_with_decay = []
    vars_without_decay = []
    for name, param in model.named_parameters():
        if 'layer_norm' in name or name.endswith('bias'):
            vars_without_decay.append(param)
        else:
            vars_with_decay.append(param)

    opt_params = [
        {
            'params': vars_with_decay,
            'weight_decay': 0.01,
        },
        {
            'params': vars_without_decay,
            'weight_decay': 0.0,
        }
    ]
    optim = BertAdam(opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, functools.partial(
            model_utils.get_lr_multiplier, total_steps=num_train_steps,
            warmup_steps=num_warmup_steps))

    train_dataset = tx.data.RecordData(hparams=config_data.train_hparam)
    eval_dataset = tx.data.RecordData(hparams=config_data.eval_hparam)
    test_dataset = tx.data.RecordData(hparams=config_data.test_hparam)

    iterator = tx.data.DataIterator(
        {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
    )

    def _train_epoch():
        """Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.switch_to_dataset("train")

        model.train()
        for batch in iterator:
            optim.zero_grad()
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                segment_ids = segment_ids.cuda()
                labels = labels.cuda()

            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, preds, loss = model(
                inputs=input_ids,
                sequence_length=input_length,
                segment_ids=segment_ids,
                labels=labels
            )
            loss.backward()
            optim.step()
            scheduler.step()
            step = scheduler.last_epoch

            dis_steps = config_data.display_steps
            if dis_steps > 0 and step % dis_steps == 0:
                logging.info(f"step: {step}; loss: {loss};")

            eval_steps = config_data.eval_steps
            if eval_steps > 0 and step % eval_steps == 0:
                _eval_epoch()

    @torch.no_grad()
    def _eval_epoch():
        """Evaluates on the dev set.
        """
        iterator.switch_to_dataset("eval")
        model.eval()
        cum_acc = 0.0
        cum_loss = 0.0
        nsamples = 0
        for batch in iterator:
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                segment_ids = segment_ids.cuda()
                labels = labels.cuda()

            batch_size = input_ids.size()[0]
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, preds, loss = model(
                inputs=input_ids,
                sequence_length=input_length,
                segment_ids=segment_ids,
                labels=labels,
            )

            accu = tx.evals.accuracy(labels, preds)
            cum_acc += accu * batch_size
            cum_loss += loss * batch_size
            nsamples += batch_size
        logging.info(
            "eval accu: {}; loss: {}; nsamples: {}".format(
                cum_acc / nsamples, cum_loss / nsamples, nsamples
            )
        )

    @torch.no_grad()
    def _test_epoch():
        """Does predictions on the test set.
        """
        iterator.switch_to_dataset("test")
        model.eval()
        _all_preds = []
        for batch in iterator:
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                segment_ids = segment_ids.cuda()

            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, preds, _ = model(
                inputs=input_ids,
                sequence_length=input_length,
                segment_ids=segment_ids,
            )
            _all_preds.extend(preds.tolist())

        output_file = os.path.join(args.output_dir, "test_results.tsv")
        with open(output_file, "w+") as writer:
            writer.write("\n".join(str(p) for p in _all_preds))

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    if args.do_train:
        for i in range(config_data.max_train_epoch):
            _train_epoch()
        states = {
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(states, os.path.join(args.output_dir + '/model.ckpt'))

    if args.do_eval:
        _eval_epoch()

    if args.do_test:
        _test_epoch()


if __name__ == "__main__":
    main()
