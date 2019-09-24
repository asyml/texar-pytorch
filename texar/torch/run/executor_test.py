import shutil
import tempfile
import unittest
from pathlib import Path
import os
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

import texar.torch as tx
from texar.torch.run import *


class DummyClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int):
        super().__init__()
        self.embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams={"dim": 10})
        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=10, hparams={
                "rnn_cell_fw": {"kwargs": {"num_units": 256}},
            })
        self.linear = nn.Linear(sum(self.encoder.output_size), n_classes)

    def _compute_logits(self, tokens: torch.LongTensor) -> torch.Tensor:
        embeds = self.embedder(tokens)
        fw_state, bw_state = self.encoder(embeds)[1]
        state = torch.cat([fw_state[0], bw_state[0]], dim=1)
        logits = self.linear(state)
        return logits

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        logits = self._compute_logits(batch.tokens)
        loss = F.cross_entropy(logits, batch.label)
        preds = torch.argmax(logits, dim=1)
        return {"loss": loss, "preds": preds}

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        logits = self._compute_logits(batch.tokens)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds}


Example = Tuple[torch.LongTensor, int]


class DummyData(tx.data.DatasetBase[Example, Example]):
    def process(self, raw_example: Example) -> Example:
        return raw_example

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        tokens = torch.stack([x for x, _ in examples], dim=0).to(self.device)
        labels = torch.tensor([x for _, x in examples], device=self.device)
        return tx.data.Batch(len(examples), tokens=tokens, label=labels)


class ExecutorTest(unittest.TestCase):
    def _create_dataset(self, n_examples: int):
        data = torch.randint(self.vocab_size, size=(n_examples, 20))
        labels = torch.randint(self.n_classes, size=(n_examples,)).tolist()
        source = tx.data.SequenceDataSource(list(zip(data, labels)))
        dataset = DummyData(source, hparams={"batch_size": 10})
        return dataset

    def setUp(self) -> None:
        make_deterministic()
        self.vocab_size = 100
        self.n_classes = 5
        self.model = DummyClassifier(self.vocab_size, self.n_classes)
        self.datasets = {
            split: self._create_dataset(n_examples)
            for split, n_examples in [
                ("train", 200), ("valid", 50), ("test1", 50), ("test2", 20)]}

        self.checkpoint_dir = tempfile.mkdtemp()
        self.tbx_logging_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.checkpoint_dir)
        shutil.rmtree(self.tbx_logging_dir)

    def test_train_loop(self):
        executor = Executor(
            model=self.model,
            train_data=self.datasets["train"],
            valid_data=self.datasets["valid"],
            test_data=[self.datasets["test1"], ("t2", self.datasets["test2"])],
            test_mode='eval',
            checkpoint_dir=self.checkpoint_dir,
            max_to_keep=3,
            save_every=[cond.time(seconds=10), cond.validation(better=True)],
            train_metrics=[("loss", metric.RunningAverage(20)),
                           metric.F1(pred_name="preds", mode="macro"),
                           metric.Accuracy(pred_name="preds")],
            optimizer={"type": torch.optim.Adam, "kwargs": {}},
            stop_training_on=cond.epoch(10),
            valid_metrics=[metric.F1(pred_name="preds", mode="micro"),
                           ("loss", metric.Average())],
            validate_every=[cond.epoch()],
            test_metrics=[metric.F1(pred_name="preds", mode="weighted")],
            plateau_condition=[
                cond.consecutive(cond.validation(better=False), 2)],
            action_on_plateau=[action.early_stop(patience=2),
                               action.reset_params(),
                               action.scale_lr(0.8)],
            log_every=cond.iteration(20),
            show_live_progress=True,
        )

        executor.train()
        executor.test()

    def test_tbx_logging(self):
        executor = Executor(
            model=self.model,
            train_data=self.datasets["train"],
            valid_data=self.datasets["valid"],
            test_data=[self.datasets["test1"], ("t2", self.datasets["test2"])],
            test_mode='eval',
            tbx_logging_dir=self.tbx_logging_dir,
            tbx_log_every=cond.iteration(1),
            checkpoint_dir=self.checkpoint_dir,
            max_to_keep=3,
            save_every=[cond.time(seconds=10), cond.validation(better=True)],
            train_metrics=[("loss", metric.RunningAverage(20)),
                           metric.F1(pred_name="preds", mode="macro"),
                           metric.Accuracy(pred_name="preds")],
            optimizer={"type": torch.optim.Adam, "kwargs": {}},
            stop_training_on=cond.epoch(10),
            valid_metrics=[metric.F1(pred_name="preds", mode="micro"),
                           ("loss", metric.Average())],
            validate_every=[cond.epoch()],
            test_metrics=[metric.F1(pred_name="preds", mode="weighted")],
            plateau_condition=[
                cond.consecutive(cond.validation(better=False), 2)],
            action_on_plateau=[action.early_stop(patience=2),
                               action.reset_params(),
                               action.scale_lr(0.8)],
            log_every=cond.iteration(20),
            show_live_progress=True,
        )

        executor.train()
        path = Path(self.tbx_logging_dir)
        self.assertTrue(path.exists())
        self.assertEqual(len(list(os.walk(path))), 1)


if __name__ == "__main__":
    test = ExecutorTest()
    try:
        test.setUp()
        test.test_train_loop()
    finally:
        test.tearDown()
