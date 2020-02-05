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
"""
Unit tests for condition related operations.
"""
import unittest

import torch

import texar.torch as tx
from texar.torch.run import condition as cond
from texar.torch.run.executor_test import DummyClassifier, DummyData


class ConditionTest(unittest.TestCase):
    def _create_dataset(self, n_examples: int):
        data = torch.randint(self.vocab_size, size=(n_examples, 20))
        labels = torch.randint(self.n_classes, size=(n_examples,)).tolist()
        source = tx.data.SequenceDataSource(list(zip(data, labels)))
        dataset = DummyData(source, hparams={"batch_size": 10})
        return dataset

    def setUp(self) -> None:
        self.vocab_size = 100
        self.n_classes = 5
        self.model = DummyClassifier(self.vocab_size, self.n_classes)
        self.datasets = {
            split: self._create_dataset(n_examples)
            for split, n_examples in [
                ("train", 200), ("valid", 50), ("test1", 50), ("test2", 20)]}

    def test_once(self):
        num_iters = 2
        max_triggers = 3

        test = self

        class Recorder:
            def __init__(self):
                self.count = 0
                self.once_triggered = False

            def every_time_fn(self, executor):
                self.count += 1
                test.assertEqual(
                    executor.status["iteration"], self.count * num_iters)
                if self.count > max_triggers:
                    test.fail("Training should terminate but didn't")
                elif self.count == max_triggers:
                    executor.terminate()

            def once_fn(self, _):
                test.assertFalse(self.once_triggered)
                self.once_triggered = True

        recorder = Recorder()

        executor = tx.run.Executor(
            model=self.model,
            train_data=self.datasets["train"],
            valid_data=self.datasets["valid"],
            test_data=[self.datasets["test1"], ("t2", self.datasets["test2"])],
            optimizer={"type": torch.optim.Adam}
        )

        executor.on(cond.iteration(num_iters), recorder.every_time_fn)
        executor.on(cond.once(cond.iteration(num_iters)), recorder.once_fn)

        executor.train()
