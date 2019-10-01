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

import argparse
import importlib
import logging
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
import texar.torch as tx
from texar.torch.run import *  # pylint: disable=wildcard-import

from utils import dataset, model_utils
from utils.processor import get_processor_class


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
        "--save-dir", type=str, default=None,
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


def construct_datasets(args) -> Dict[str, tx.data.RecordData]:
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
        shuffle_buffer=None)
    return datasets


class RegressorWrapper(tx.modules.XLNetRegressor):
    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        preds = super().forward(inputs=batch.input_ids,
                                segment_ids=batch.segment_ids,
                                input_mask=batch.input_mask)
        loss = (preds - batch.label_ids) ** 2
        loss = loss.sum() / len(batch)
        return {"loss": loss, "preds": preds}


class ClassifierWrapper(tx.modules.XLNetClassifier):
    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        logits, preds = super().forward(inputs=batch.input_ids,
                                        segment_ids=batch.segment_ids,
                                        input_mask=batch.input_mask)
        loss = F.cross_entropy(logits, batch.label_ids, reduction='none')
        loss = loss.sum() / len(batch)
        return {"loss": loss, "preds": preds}


def main(args) -> None:
    if args.seed != -1:
        make_deterministic(args.seed)
        print(f"Random seed set to {args.seed}")

    datasets = construct_datasets(args)
    print("Dataset constructed")

    processor_class = get_processor_class(args.task)
    is_regression = processor_class.is_regression
    model: Union[RegressorWrapper, ClassifierWrapper]
    if is_regression:
        model = RegressorWrapper(
            pretrained_model_name=args.pretrained_model_name)
    else:
        model = ClassifierWrapper(
            pretrained_model_name=args.pretrained_model_name,
            hparams={"num_classes": len(processor_class.labels)})
    print("Model constructed")

    optim = torch.optim.Adam(
        model.param_groups(args.lr, args.lr_layer_decay_rate), lr=args.lr,
        eps=args.adam_eps, weight_decay=args.weight_decay)
    lambda_lr = model_utils.warmup_lr_lambda(
        args.train_steps, args.warmup_steps, args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda_lr)

    bps = args.backwards_per_step

    def get_condition(steps: int) -> Optional[cond.Condition]:
        if steps == -1:
            return None
        return cond.iteration(steps * bps)

    if is_regression:
        valid_metric: metric.Metric = metric.PearsonR(
            pred_name="preds", label_name="label_ids")
    else:
        valid_metric = metric.Accuracy(
            pred_name="preds", label_name="label_ids")
    executor = Executor(
        model=model,
        train_data=datasets["train"],
        valid_data=datasets["dev"],
        test_data=datasets.get("test", None),
        checkpoint_dir=args.save_dir or f"saved_models/{args.task}",
        save_every=get_condition(args.save_steps),
        max_to_keep=1,
        train_metrics=[
            ("loss", metric.RunningAverage(args.display_steps * bps)),
            metric.LR(optim)],
        optimizer=optim,
        lr_scheduler=scheduler,
        grad_clip=args.grad_clip,
        num_iters_per_update=args.backwards_per_step,
        log_every=cond.iteration(args.display_steps * bps),
        validate_every=get_condition(args.eval_steps),
        valid_metrics=[valid_metric, ("loss", metric.Average())],
        stop_training_on=cond.iteration(args.train_steps * bps),
        log_format="{time} : Epoch {epoch} @ {iteration:5d}it "
                   "({speed}), LR = {LR:.3e}, loss = {loss:.3f}",
        test_mode='eval',
        show_live_progress=True,
    )

    if args.checkpoint is not None:
        executor.load(args.checkpoint)

    if args.mode == 'train':
        executor.train()
        executor.save()
        executor.test(tx.utils.dict_fetch(datasets, ["dev", "test"]))
    else:
        if args.checkpoint is None:
            executor.load(load_training_state=False)  # load previous best model
        executor.test(tx.utils.dict_fetch(datasets, ["dev", "test"]))


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    _args = parse_args()
    main(_args)
