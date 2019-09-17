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
import functools
import importlib
import sys
import logging
import shutil
from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

import hyperopt as hpo

import texar.torch as tx
from texar.torch.run import *
from texar.torch.modules import BERTClassifier

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
args = parser.parse_args()

config_data: Any = importlib.import_module(args.config_data)
config_downstream = importlib.import_module(args.config_downstream)
config_downstream = {
    k: v for k, v in config_downstream.__dict__.items()
    if not k.startswith('__')}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.root.setLevel(logging.INFO)


class ModelWrapper(nn.Module):
    r"""This class wraps a model (in this case a BERT classifier) and implements
    :meth:`forward` and :meth:`predict` to conform to the requirements of
    :class:`texar.torch.run.Executor` class. Particularly, :meth:`forward`
    returns a dict with keys "loss" and "preds" and :meth:`predict` returns a
    dict with key "preds".

    Args:
        `model`: BERTClassifier
            A BERTClassifier model
    """

    def __init__(self, model: BERTClassifier):
        super().__init__()
        self.model = model

    def _compute_loss(self, logits, labels):
        if self.model.is_binary:
            loss = F.binary_cross_entropy(
                logits.view(-1), labels.view(-1), reduction='mean')
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.model.num_classes),
                labels.view(-1), reduction='mean')
        return loss

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        r"""Run forward through the model and return a dict to be consumed
        by the :class:`texar.torch.run.Executor`. This method will be called by
        :class:`texar.torch.run.Executor` during training. See
        https://texar-pytorch.readthedocs.io/en/latest/code/run.html#executor-general-args
        for more details.

        Args:
            `batch`: :class:`texar.data.Batch`. (See
                https://texar-pytorch.readthedocs.io/en/latest/code/data.html#texar.torch.data.Batch
                for more details)
                A batch of inputs to be passed through the model

        Returns:
            A dict with keys "loss" and "preds" containing the loss and
            predictions on :attr:`batch` respectively.
        """
        input_ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        labels = batch["label_ids"]

        input_length = (1 - (input_ids == 0).int()).sum(dim=1)

        logits, preds = self.model(input_ids, input_length, segment_ids)

        loss = self._compute_loss(logits, labels)

        return {"loss": loss, "preds": preds}

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        r"""Predict the labels for the :attr:`batch` of examples. This method
        will be called instead of :meth:`forward` during validation or testing,
        if :class:`texar.torch.run.Executor`'s :attr:`validate_mode` or
        :attr:`test_mode` is set to ``"predict"`` instead of ``"eval"``.

        Args:
            `batch`: tx.data.Batch
                A batch of inputs to run prediction on
        """
        input_ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]

        input_length = (1 - (input_ids == 0).int()).sum(dim=1)

        _, preds = self.model(input_ids, input_length, segment_ids)

        return {"preds": preds}


class TPE:
    r""":class:`TPE` uses Tree-structured Parzen Estimator algorithm from
    `hyperopt` for hyperparameter tuning.

    Args:
        model_config: Dict
            A conf dict which is passed to BERT classifier
        output_dir: str
            A path to store the models
    """

    def __init__(self, model_config: Dict, output_dir: str = "output/"):
        tx.utils.maybe_create_dir(output_dir)

        self.model_config = model_config

        self.output_dir = output_dir

        # create datasets
        self.train_dataset = tx.data.RecordData(
            hparams=config_data.train_hparam, device=device)
        self.eval_dataset = tx.data.RecordData(
            hparams=config_data.eval_hparam, device=device)

        # Builds BERT
        model = tx.modules.BERTClassifier(
            pretrained_model_name=args.pretrained_model_name,
            hparams=self.model_config)
        self.model = ModelWrapper(model=model)
        self.model.to(device)

        # batching
        self.batching_strategy = \
            tx.data.TokenCountBatchingStrategy[Dict[str, Any]](
                max_tokens=config_data.max_batch_tokens)

        # logging formats
        self.log_format = "{time} : Epoch {epoch:2d} @ {iteration:6d}it " \
                          "({progress}%, {speed}), " \
                          "lr = {lr:.9e}, loss = {loss:.3f}"
        self.valid_log_format = "{time} : Epoch {epoch}, " \
                                "{split} accuracy = {Accuracy:.3f}, " \
                                "loss = {loss:.3f}"
        self.valid_progress_log_format = "{time} : Evaluating on " \
                                         "{split} ({progress}%, {speed})"

        # exp number
        self.exp_number = 1

    def objective_func(self, hyperparams: Dict):
        r"""Compute "loss" for a given hyperparameter values. This function is
        passed to hyperopt's ``"fmin"`` (see the :meth:`run` method) function
        and gets repeatedly called to find the best set of hyperparam values.
        Below is an example of how to use this method

        .. code-block:: python

            import hyperopt as hpo

            trials = hpo.Trials()
            hpo.fmin(fn=self.objective_func,
                     space=space,
                     algo=hpo.tpe.suggest,
                     max_evals=3,
                     trials=trials)

        Args:
            hyperparams: Dict
                A `(key, value)` dict representing the ``"value"`` to try for
                the hyperparam ``"key"``

        Returns:
            A dict with keys "loss", "status" and "model" indicating the loss
            for this trial, the status, and the path to the saved model.
        """
        print(f"Using {hyperparams} for trial {self.exp_number}")

        # Loads data
        num_train_data = config_data.num_train_data
        num_train_steps = int(num_train_data / config_data.train_batch_size *
                              config_data.max_train_epoch)

        # hyperparams
        num_warmup_steps = hyperparams["optimizer.warmup_steps"]
        static_lr = hyperparams["optimizer.static_lr"]

        vars_with_decay = []
        vars_without_decay = []
        for name, param in self.model.named_parameters():
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

        valid_metric = metric.Accuracy[float](
            pred_name="preds", label_name="label_ids")
        checkpoint_dir = f"./{self.output_dir}/exp{self.exp_number}"
        log_file = f"./{self.output_dir}/log.txt"

        executor = Executor(
            # supply executor with the model
            model=self.model,
            # define datasets
            train_data=self.train_dataset,
            valid_data=self.eval_dataset,
            batching_strategy=self.batching_strategy,
            device=device,
            # training and stopping details
            optimizer=optim,
            lr_scheduler=scheduler,
            stop_training_on=cond.epoch(config_data.max_train_epoch),
            # logging details
            log_every=[cond.epoch(1)],
            log_destination=[sys.stdout, log_file],
            # logging format
            log_format=self.log_format,
            # define metrics
            train_metrics=[
                ("loss", metric.RunningAverage(1)),
                ("lr", metric.LR(optim))],
            valid_metrics=[valid_metric, ("loss", metric.Average())],
            validate_every=cond.epoch(1),
            save_every=cond.epoch(config_data.max_train_epoch),
            checkpoint_dir=checkpoint_dir,
            max_to_keep=1,
            show_live_progress=True,
            print_model_arch=False
        )

        if args.checkpoint is not None:
            executor.load(args.checkpoint)

        executor.train()

        print(f"Loss on the valid dataset "
              f"{executor.valid_metrics['loss'].value()}")
        self.exp_number += 1

        return {
            "loss": executor.valid_metrics["loss"].value(),
            "status": hpo.STATUS_OK,
            "model": checkpoint_dir
        }

    def run(self, hyperparams: Dict):
        r"""Run the TPE algorithm with hyperparameters  :attr:`hyperparams`

        Args:
            hyperparams: Dict
                The `(key, value)` pairs of hyperparameters along their range of
                values.
        """
        space = {}
        for k, v in hyperparams.items():
            if isinstance(v, dict):
                if v["dtype"] == int:
                    space[k] = hpo.hp.choice(
                        k, range(v["start"], v["end"]))
                else:
                    space[k] = hpo.hp.uniform(k, v["start"], v["end"])
        trials = hpo.Trials()
        hpo.fmin(fn=self.objective_func,
                 space=space,
                 algo=hpo.tpe.suggest,
                 max_evals=3,
                 trials=trials)
        _, best_trial = min((trial["result"]["loss"], trial)
                            for trial in trials.trials)

        # delete all the other models
        for trial in trials.trials:
            if trial is not best_trial:
                shutil.rmtree(trial["result"]["model"])


def main():
    model_config = {k: v for k, v in config_downstream.items() if
                    k != "hyperparams"}
    tpe = TPE(model_config=model_config, output_dir=args.output_dir)
    hyperparams = config_downstream["hyperparams"]
    tpe.run(hyperparams)


if __name__ == '__main__':
    main()
