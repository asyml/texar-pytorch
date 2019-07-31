import shutil
import tempfile
import unittest
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

import texar.torch as tx
from texar.torch.run import *


class TestModel(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int):
        super().__init__()
        self.embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams={"dim": 10})
        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=10, hparams={
                "rnn_cell_fw": {"kwargs": {"num_units": 256}},
            })
        self.linear = nn.Linear(sum(self.encoder.output_size), n_classes)

    def forward(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        embeds = self.embedder(batch.tokens)
        fw_state, bw_state = self.encoder(embeds)[1]
        state = torch.cat([fw_state[0], bw_state[0]], dim=1)
        logits = self.linear(state)
        loss = F.cross_entropy(logits, batch.label)
        preds = torch.argmax(logits, dim=1)
        return {"loss": loss, "preds": preds}


Example = Tuple[torch.LongTensor, int]


class DummyData(tx.data.DataBase[Example, Example]):
    def process(self, raw_example: Example) -> Example:
        return raw_example

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        tokens = torch.stack([x for x, _ in examples], dim=0).to(self.device)
        labels = torch.tensor([x for _, x in examples], device=self.device)
        return tx.data.Batch(len(examples), tokens=tokens, label=labels)


class ExecutorTest(unittest.TestCase):
    def _create_dataset(self, n_examples: int):
        data = torch.randint(self.vocab_size, size=(n_examples, 20))
        labels = torch.randint(self.n_classes, size=(n_examples,))
        source = tx.data.SequenceDataSource(list(zip(data, labels)))
        dataset = DummyData(source, hparams={"batch_size": 10})
        return dataset

    def setUp(self) -> None:
        self.vocab_size = 100
        self.n_classes = 5
        self.model = TestModel(self.vocab_size, self.n_classes)
        self.datasets = {
            split: self._create_dataset(n_examples)
            for split, n_examples in [
                ("train", 1000), ("valid", 100), ("test", 100)]}

        self.checkpoint_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.checkpoint_dir)

    def test_train_loop(self):
        make_deterministic()

        executor = Executor(
            model=self.model,
            train_data=self.datasets["train"],
            valid_data=self.datasets["valid"],
            test_data=self.datasets["test"],
            checkpoint_dir=self.checkpoint_dir,
            max_to_keep=3,
            save_every=[cond.time(seconds=50), cond.validation(better=True)],
            train_metrics=[("loss", metric.RunningAverage(20)),
                           metric.F1(pred_name="preds"),
                           metric.Accuracy(pred_name="preds")],
            optimizer={"type": torch.optim.Adam, "kwargs": {}},
            max_epochs=20,
            valid_metrics=[metric.F1(pred_name="preds"),
                           ("loss", metric.Average())],
            validate_every=[cond.epoch()],
            plateau_condition=[
                cond.consecutive(cond.validation(better=False), 2)],
            action_on_plateau=[action.early_stop(patience=2),
                               action.reset_params(),
                               action.scale_lr(0.8)],
            log_every=cond.iteration(20),
        )

        executor.train()


if __name__ == "__main__":
    test = ExecutorTest()
    try:
        test.setUp()
        test.test_train_loop()
    finally:
        test.tearDown()
