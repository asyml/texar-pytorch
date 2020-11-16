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
"""Example of building a sentence classifier based on pre-trained BERT model.
"""

import argparse
import functools
import importlib
import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import adaptdl
from utils import model_utils
import nni

import texar.torch as tx
import texar.torch.distributed  # pylint: disable=unused-import

IS_CHIEF = int(os.getenv("ADAPTDL_RANK", "0")) == 0

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
    help="Path to a model checkpoint (including bert modules) to restore from")
parser.add_argument(
    "--do-train", action="store_true", default=True,
    help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true", default=True,
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
tuner_params = nni.get_next_parameter()

logging.root.setLevel(logging.INFO)

# Initialize process group with distributed training backend.
# adaptdl.readthedocs.io/en/latest/adaptdl-pytorch.html#initializing-adaptdl
adaptdl.torch.init_process_group("nccl" if
            torch.cuda.is_available() else "gloo")

# Get the shared path on distributed shared storage
if adaptdl.env.share_path():  # Will be set by the AdaptDL controller
    OUTPUT_DIR = adaptdl.env.share_path()
else:
    OUTPUT_DIR = args.output_dir


tensorboard_dir = os.path.join(
    os.getenv("ADAPTDLCTL_TENSORBOARD_LOGDIR", "/adaptdl/tensorboard"),
    os.getenv("NNI_TRIAL_JOB_ID", "lr-adaptdl")
)


def main() -> None:
    """
    Builds the model and runs.
    """
    # Loads data
    num_train_data = config_data.num_train_data

    # Builds BERT
    model = tx.modules.BERTClassifier(
        pretrained_model_name=args.pretrained_model_name,
        hparams=config_downstream)
    model.to(device)

    num_train_steps = int(num_train_data / config_data.train_batch_size *
                          config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)

    # Builds learning rate decay scheduler
    static_lr = tuner_params['static_lr']

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

    iterator = tx.distributed.AdaptiveDataIterator(
        {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
    )

    # We wrap the model in AdaptiveDataParallel to make it distributed
    model = tx.distributed.AdaptiveDataParallel(model, optim, scheduler)

    def _compute_loss(logits, labels):
        r"""Compute loss.
        """
        if model.is_binary:
            loss = F.binary_cross_entropy(
                logits.view(-1), labels.view(-1), reduction='mean')
        else:
            loss = F.cross_entropy(
                logits.view(-1, model.num_classes),
                labels.view(-1), reduction='mean')
        return loss

    def _train_epoch(epoch):
        r"""Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.switch_to_dataset("train")
        model.train()
        stats = adaptdl.torch.Accumulator()

        for batch in iterator:
            optim.zero_grad()
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]

            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, _ = model(input_ids, input_length, segment_ids)

            loss = _compute_loss(logits, labels)
            loss.backward()
            optim.step()
            scheduler.step()
            step = scheduler.last_epoch

            dis_steps = config_data.display_steps
            if dis_steps > 0 and step % dis_steps == 0:
                logging.info("epoch: %d, step: %d, loss: %.4f",
                             epoch, step, loss)

            gain = model.gain
            batchsize = iterator.current_batch_size
            writer.add_scalar("Throughput/Gain", gain, epoch)
            writer.add_scalar("Throughput/Global_Batchsize",
                              batchsize, epoch)
            stats["loss_sum"] += loss.item() * labels.size(0)
            stats["total"] += labels.size(0)

        with stats.synchronized():
            stats["loss_avg"] = stats["loss_sum"] / stats["total"]
            writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)


    @torch.no_grad()
    def _eval_epoch(epoch):
        """Evaluates on the dev set.
        """
        iterator.switch_to_dataset("eval")
        model.eval()
        stats = adaptdl.torch.Accumulator()

        nsamples = 0
        avg_rec = tx.utils.AverageRecorder()
        for batch in iterator:
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]

            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, preds = model(input_ids, input_length, segment_ids)

            loss = _compute_loss(logits, labels)
            accu = tx.evals.accuracy(labels, preds)
            batch_size = input_ids.size()[0]
            avg_rec.add([accu, loss], batch_size)
            nsamples += batch_size
            stats["loss_sum"] += loss.item() * labels.size(0)
            stats["total"] += labels.size(0)

        logging.info("eval accu: %.4f; loss: %.4f; nsamples: %d",
                     avg_rec.avg(0), avg_rec.avg(1), nsamples)
        with stats.synchronized():
            stats["loss_avg"] = stats["loss_sum"] / stats["total"]
            writer.add_scalar("Loss/Validation", stats["loss_avg"], epoch)
            if IS_CHIEF:
                if epoch < config_data.max_train_epoch - 1:
                    nni.report_intermediate_result(stats['loss_avg'], 
                        accum=stats)
                else:
                    nni.report_final_result(stats['loss_avg'])



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

            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            _, preds = model(input_ids, input_length, segment_ids)

            _all_preds.extend(preds.tolist())

        if adaptdl.env.replica_rank() == 0:
            # Only allow writes by the main replica
            output_file = os.path.join(OUTPUT_DIR, "test_results.tsv")
            with open(output_file, "w+") as writer:
                writer.write("\n".join(str(p) for p in _all_preds))
            logging.info("test output written to %s", output_file)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    with SummaryWriter(tensorboard_dir) as writer:
        if args.do_train:
            # We get remaining epochs from AdaptDL in a restart-safe way.
            for epoch in adaptdl.torch.remaining_epochs_until(
                                         config_data.max_train_epoch):
                _train_epoch(epoch)

                if args.do_eval:
                    _eval_epoch(epoch)

            states = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
            }

            if adaptdl.env.replica_rank() == 0:
                # Only allow writes by the main replica
                torch.save(states, os.path.join(OUTPUT_DIR, 'model.ckpt'))




    if args.do_test:
        _test_epoch()


if __name__ == "__main__":
    main()
