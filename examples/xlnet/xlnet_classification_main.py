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

import argparse
import importlib
import logging
import os
import random
import time
from typing import Optional, Dict

import numpy as np
import torch
import tqdm
import sentencepiece as spm
import texar as tx

import xlnet


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
        "--pretrained", type=str,
        default="pretrained/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt",
        help="Path to the pretrained XLNet model checkpoint")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the directory containing raw data. "
             "Defaults to 'data/<task name>'")
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Path to the directory to cache processed data. "
             "Defaults to 'processed_data/<task name>'")
    parser.add_argument(
        "--spm-model-path", type=str,
        default="pretrained/xlnet_cased_L-24_H-1024_A-16/spiece.model",
        help="Path to the sentencepiece model")
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
    metric: xlnet.model.StreamingMetric
    if is_regression:
        metric = xlnet.model.StreamingPearsonR()
    else:
        metric = xlnet.model.StreamingAccuracy()
    avg_loss = tx.utils.AverageRecorder()
    progress = tqdm.tqdm(iterator, ncols=80, **(tqdm_kwargs or {}))
    for batch in progress:
        loss, logits = model(
            batch.input_ids.t(), batch.segment_ids.t(),
            batch.label_ids, batch.input_mask.t())
        gold_labels = batch.label_ids.view(-1).tolist()
        if is_regression:
            pred_labels = logits.tolist()
        else:
            pred_labels = torch.argmax(logits, dim=1).tolist()
        metric.add(gold_labels, pred_labels)
        avg_loss.add(loss.mean().item())
        progress.set_postfix({metric.name: f"{metric.value():.4f}"})
    print_fn(f"{metric.name.capitalize()}: {metric.value()}, "
             f"loss: {avg_loss.avg():.4f}")


def construct_datasets(args, device: Optional[torch.device] = None) \
        -> Dict[str, tx.data.RecordData]:
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(args.spm_model_path)

    cache_prefix = f"length{args.max_seq_len}"
    tokenize_fn = xlnet.data.create_tokenize_fn(sp_model, args.uncased)
    processor_class = xlnet.data.get_processor_class(args.task)
    data_dir = args.data_dir or f"data/{processor_class.task_name}"
    cache_dir = args.cache_dir or f"processed_data/{processor_class.task_name}"
    task_processor = processor_class(data_dir)
    xlnet.data.construct_dataset(
        task_processor, cache_dir, args.max_seq_len,
        tokenize_fn, file_prefix=cache_prefix)

    datasets = xlnet.data.load_datasets(
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

    processor_class = xlnet.data.get_processor_class(args.task)
    is_regression = processor_class.is_regression
    if is_regression:
        model = xlnet.model.XLNetRegressor()
    else:
        model = xlnet.model.XLNetClassifier(
            hparams={"num_classes": len(processor_class.labels)})
    model.apply(xlnet.model.init_weights)
    print("Weights initialized")

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        xlnet.model.load_from_tf_checkpoint(model, args.pretrained)
        print(f"Loaded pretrained weights from {args.pretrained}")
    model = model.to(device)
    print("Model constructed")
    print("Model structure:", model)

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
    lambda_lr = xlnet.model.warmup_lr_lambda(
        args.train_steps, args.warmup_steps, args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda_lr)

    avg_loss = tx.utils.AverageRecorder()
    train_steps = 0
    grad_steps = 0
    total_batch_size = args.batch_size * args.backwards_per_step
    progress = tqdm.tqdm(xlnet.data.repeat(
        lambda: iterator.get_iterator('train')), ncols=80)
    for batch in progress:
        model.train()
        loss, _ = model(
            batch.input_ids.t(), batch.segment_ids.t(),
            batch.label_ids, batch.input_mask.t())
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
