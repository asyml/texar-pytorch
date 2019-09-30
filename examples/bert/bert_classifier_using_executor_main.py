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
"""Example of building a sentence classifier on top of pre-trained BERT using
Texar's Executor.
"""

import argparse
import functools
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

import texar.torch as tx
from texar.torch.run import *

from utils import model_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-downstream", default="config_classifier",
    help="Configuration of the downstream part of the model")
parser.add_argument(
    '--pretrained-model-name', type=str, default='bert-base-uncased',
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained checkpoint to load.")
parser.add_argument(
    "--config-data", default="config_data", help="The dataset config.")
parser.add_argument(
    "--output-dir", default="output/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a model checkpoint (including bert modules) to restore from.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do-test", action="store_true",
    help="Whether to run test on the test set.")
args = parser.parse_args()

config_data: Any = importlib.import_module(args.config_data)
config_downstream = importlib.import_module(args.config_downstream)
config_downstream = {
    k: v for k, v in config_downstream.__dict__.items()
    if not k.startswith('__') and k != "hyperparams"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelWrapper(nn.Module):
    def __init__(self, model: tx.modules.BERTClassifier):
        super().__init__()
        self.model = model

    def _get_outputs(self, batch: tx.data.Batch) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        input_ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        input_length = (1 - (input_ids == 0).int()).sum(dim=1)
        logits, preds = self.model(input_ids, input_length, segment_ids)
        return logits, preds

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        logits, preds = self._get_outputs(batch)
        labels = batch["label_ids"]
        if self.model.is_binary:
            loss = F.binary_cross_entropy(
                logits.view(-1), labels.view(-1), reduction='mean')
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.model.num_classes),
                labels.view(-1), reduction='mean')
        return {"loss": loss, "preds": preds}

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        _, preds = self._get_outputs(batch)
        return {"preds": preds}


class FileWriterMetric(metric.SimpleMetric[List[int], float]):
    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        super().__init__(pred_name="preds", label_name="input_ids")
        self.file_path = file_path

    def _value(self) -> float:
        path = self.file_path or tempfile.mktemp()
        with open(path, "w+") as writer:
            writer.write("\n".join(str(p) for p in self.predicted))
        return 1.0


def main() -> None:
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(args.output_dir)

    # Loads data
    num_train_data = config_data.num_train_data

    # Builds BERT
    model = tx.modules.BERTClassifier(
        pretrained_model_name=args.pretrained_model_name,
        hparams=config_downstream)
    model = ModelWrapper(model=model)

    num_train_steps = int(num_train_data / config_data.train_batch_size *
                          config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)

    # Builds learning rate decay scheduler
    static_lr = 2e-5

    vars_with_decay = []
    vars_without_decay = []
    for name, param in model.named_parameters():
        if 'layer_norm' in name or name.endswith('bias'):
            vars_without_decay.append(param)
        else:
            vars_with_decay.append(param)

    opt_params = [{
        'params': vars_with_decay,
        'weight_decay': 0.01,
    }, {
        'params': vars_without_decay,
        'weight_decay': 0.0,
    }]
    optim = tx.core.BertAdam(
        opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, functools.partial(model_utils.get_lr_multiplier,
                                 total_steps=num_train_steps,
                                 warmup_steps=num_warmup_steps))

    train_dataset = tx.data.RecordData(hparams=config_data.train_hparam,
                                       device=device)
    eval_dataset = tx.data.RecordData(hparams=config_data.eval_hparam,
                                      device=device)
    test_dataset = tx.data.RecordData(hparams=config_data.test_hparam,
                                      device=device)

    batching_strategy = tx.data.TokenCountBatchingStrategy[Dict[str, Any]](
        max_tokens=config_data.max_batch_tokens)

    output_dir = Path(args.output_dir)
    valid_metric = metric.Accuracy[float](
        pred_name="preds", label_name="label_ids")

    executor = Executor(
        # supply executor with the model
        model=model,
        # define datasets
        train_data=train_dataset,
        valid_data=eval_dataset,
        test_data=test_dataset,
        batching_strategy=batching_strategy,
        device=device,
        # tbx logging
        tbx_logging_dir=os.path.join(config_data.tbx_log_dir,
                                     "exp" + str(config_data.exp_number)),
        tbx_log_every=cond.iteration(config_data.tbx_logging_steps),
        # training and stopping details
        optimizer=optim,
        lr_scheduler=scheduler,
        stop_training_on=cond.epoch(config_data.max_train_epoch),
        # logging details
        log_destination=[sys.stdout, output_dir / "log.txt"],
        log_every=[cond.iteration(config_data.display_steps)],
        # logging format
        log_format="{time} : Epoch {epoch:2d} @ {iteration:6d}it "
                   "({progress}%, {speed}), lr = {lr:.3e}, loss = {loss:.3f}",
        valid_log_format="{time} : Epoch {epoch}, "
                         "{split} accuracy = {Accuracy:.3f}, loss = {loss:.3f}",
        valid_progress_log_format="{time} : Evaluating on "
                                  "{split} ({progress}%, {speed})",
        test_log_format="{time} : Epoch {epoch}, "
                        "{split} accuracy = {Accuracy:.3f}",
        # define metrics
        train_metrics=[
            ("loss", metric.RunningAverage(1)),  # only show current loss
            ("lr", metric.LR(optim))],
        valid_metrics=[valid_metric, ("loss", metric.Average())],
        test_metrics=[
            valid_metric, FileWriterMetric(output_dir / "test.output")],
        # freq of validation
        validate_every=[cond.iteration(config_data.eval_steps)],
        # checkpoint saving location
        checkpoint_dir=args.output_dir,
        save_every=cond.validation(better=True),
        test_mode='predict',
        max_to_keep=1,
        show_live_progress=True,
    )

    if args.checkpoint is not None:
        executor.load(args.checkpoint)

    if args.do_train:
        executor.train()

    if args.do_test:
        executor.test()


if __name__ == "__main__":
    main()
