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
"""Example of building XLNet language model for classification/regression.
"""

from typing import Optional, Dict

import argparse
import importlib
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import texar.torch as tx

# pylint: disable=wildcard-import

from utils.processor import *
from utils.processor import get_processor_class
from utils import data_utils, dataset, metrics, model_utils


def load_config_into_args(config_path: str, args):
    config_module_path = config_path.replace('/', '.').replace('\\', '.')
    if config_module_path.endswith(".py"):
        config_module_path = config_module_path[:-3]
    config_data = importlib.import_module(config_module_path)
    for key in dir(config_data):
        if not key.startswith('__'):
            setattr(args, key, getattr(config_data, key))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-data", required=True,
        help="Path to the dataset configuration file.")

    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a saved checkpoint file to load")
    parser.add_argument(
        "--save-dir", type=str, default="saved_models/",
        help="Directory to save model checkpoints")

    parser.add_argument(
        "--pretrained-model-name", type=str,
        default="xlnet-large-cased",
        help="The pre-trained model name to load selected in the list of: "
             "`xlnet-base-cased`, `xlnet-large-cased`.")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the directory containing raw data. "
             "Defaults to 'data/<task name>'")
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Path to the directory to cache processed data. "
             "Defaults to 'processed_data/<task name>'")
    parser.add_argument(
        "--uncased", type=bool, default=False,
        help="Whether the pretrained model is an uncased model")

    args = parser.parse_args()
    load_config_into_args(args.config_data, args)
    return args


@torch.no_grad()
def evaluate(model, iterator, is_regression: bool = False, print_fn=None,
             tqdm_kwargs=None):
    if print_fn is None:
        print_fn = print
    if is_regression:
        metric: metrics.StreamingMetric = metrics.StreamingPearsonR()
    else:
        metric = metrics.StreamingAccuracy()
    avg_loss = tx.utils.AverageRecorder()
    progress = tqdm.tqdm(iterator, ncols=80, **(tqdm_kwargs or {}))
    for batch in progress:
        labels = batch.label_ids
        if is_regression:
            preds = model(token_ids=batch.input_ids,
                          segment_ids=batch.segment_ids,
                          input_mask=batch.input_mask)
            loss = (preds - labels.view(-1)) ** 2
        else:
            logits, preds = model(token_ids=batch.input_ids,
                                  segment_ids=batch.segment_ids,
                                  input_mask=batch.input_mask)
            loss = F.cross_entropy(logits, labels.view(-1), reduction='none')

        gold_labels = labels.view(-1).tolist()
        pred_labels = preds.tolist()
        metric.add(gold_labels, pred_labels)
        avg_loss.add(loss.mean().item())
        progress.set_postfix({metric.name: f"{metric.value():.4f}"})
    print_fn(f"{metric.name.capitalize()}: {metric.value()}, "
             f"loss: {avg_loss.avg():.4f}")


def construct_datasets(args, device: Optional[torch.device] = None) \
        -> Dict[str, tx.data.RecordData]:

    cache_prefix = f"length{args.max_seq_len}"

    tokenizer = tx.data.XLNetTokenizer(
        pretrained_model_name=args.pretrained_model_name)
    tokenizer.do_lower_case = args.uncased

    processor_class = get_processor_class(args.task)
    data_dir = args.data_dir or f"data/{processor_class.task_name}"
    cache_dir = args.cache_dir or f"processed_data/{processor_class.task_name}"
    task_processor = processor_class(data_dir)
    dataset.construct_dataset(
        task_processor, cache_dir, args.max_seq_len,
        tokenizer, file_prefix=cache_prefix)

    datasets = dataset.load_datasets(
        args.task, cache_dir, args.max_seq_len, args.batch_size,
        file_prefix=cache_prefix, eval_batch_size=args.eval_batch_size,
        shuffle_buffer=None, device=device)
    return datasets


def main(args):
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using CUDA device {device}")
    else:
        device = 'cpu'
        print("Using CPU")
    device = torch.device(device)

    datasets = construct_datasets(args, device)
    iterator = tx.data.DataIterator(datasets)
    print("Dataset constructed")

    processor_class = get_processor_class(args.task)
    is_regression = processor_class.is_regression
    if is_regression:
        model = tx.modules.XLNetRegressor(
            pretrained_model_name=args.pretrained_model_name)
    else:
        model = tx.modules.XLNetClassifier(
            pretrained_model_name=args.pretrained_model_name,
            hparams={"num_classes": len(processor_class.labels)})
    print("Weights initialized")

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")

    model = model.to(device)
    print("Model constructed")

    def eval_all_splits():
        model.eval()
        for split in datasets:
            if split != 'train':
                print(f"Evaluating on {split}")
                evaluate(model, iterator.get_iterator(split), is_regression)

    def save_model(step: int, model):
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_name = (f"{args.task}_step{step}_"
                     f"{time.strftime('%Y%m%d_%H%M%S')}")
        save_path = os.path.join(args.save_dir, save_name)
        torch.save(model.state_dict(), save_path)
        progress.write(f"Model at {step} steps saved to {save_path}")

    if args.mode == 'eval':
        eval_all_splits()
        return

    optim = torch.optim.Adam(
        model.param_groups(args.lr, args.lr_layer_decay_rate), lr=args.lr,
        eps=args.adam_eps, weight_decay=args.weight_decay)
    lambda_lr = model_utils.warmup_lr_lambda(
        args.train_steps, args.warmup_steps, args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda_lr)

    avg_loss = tx.utils.AverageRecorder()
    train_steps = 0
    grad_steps = 0
    total_batch_size = args.batch_size * args.backwards_per_step
    progress = tqdm.tqdm(data_utils.repeat(
        lambda: iterator.get_iterator('train')), ncols=80)
    for batch in progress:
        model.train()
        labels = batch.label_ids
        if is_regression:
            preds = model(token_ids=batch.input_ids,
                          segment_ids=batch.segment_ids,
                          input_mask=batch.input_mask)
            loss = (preds - labels.view(-1)) ** 2
        else:
            logits, _ = model(token_ids=batch.input_ids,
                              segment_ids=batch.segment_ids,
                              input_mask=batch.input_mask)
            loss = F.cross_entropy(logits, labels.view(-1), reduction='none')
        loss = loss.sum() / total_batch_size
        avg_loss.add(loss.item() * args.backwards_per_step)
        loss.backward()
        grad_steps += 1
        if grad_steps == args.backwards_per_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            optim.zero_grad()
            train_steps += 1
            grad_steps = 0

            if train_steps % args.display_steps == 0:
                progress.write(
                    f"Step: {train_steps}, "
                    f"LR = {optim.param_groups[0]['lr']:.3e}, "
                    f"loss = {avg_loss.avg():.4f}")
                avg_loss.reset()

            scheduler.step()

            if train_steps >= args.train_steps:
                # Break before save & eval since we're doing them anyway.
                break

            if args.save_steps != -1 and train_steps % args.save_steps == 0:
                save_model(train_steps, model)

            if args.eval_steps != -1 and train_steps % args.eval_steps == 0:
                model.eval()
                evaluate(
                    model, iterator.get_iterator('dev'), is_regression,
                    print_fn=progress.write, tqdm_kwargs={"leave": False})
    progress.close()

    save_model(args.train_steps, model)
    eval_all_splits()


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    _args = parse_args()
    main(_args)
