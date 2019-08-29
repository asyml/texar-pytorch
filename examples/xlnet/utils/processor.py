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
Data processors. Adapted from
https://github.com/zihangdai/xlnet/blob/master/run_classifier.py
"""

import csv
import logging
from abc import ABC
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict, Type


class InputExample(NamedTuple):
    r"""A single training/test example for simple sequence classification."""
    guid: str
    r"""Unique id for the example."""
    text_a: str
    r"""string. The untokenized text of the first sequence. For single sequence
    tasks, only this sequence must be specified."""
    text_b: Optional[str]
    r"""(Optional) string. The untokenized text of the second sequence. Only
    needs to be specified for sequence pair tasks."""
    label: Optional[Union[str, float]]
    r"""(Optional) string. The label of the example. This should be specified
    for train and dev examples, but not for test examples."""


class DataProcessor:
    r"""Base class for data converters for sequence classification data sets."""
    labels: List[str]
    is_regression: bool = False

    task_name: str
    __task_dict__: Dict[str, Type['DataProcessor']] = {}

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    @classmethod
    def register(cls, *names):
        def decorator(klass):
            for name in names:
                prev_processor = DataProcessor.__task_dict__.get(
                    name.lower(), None)
                if prev_processor is not None:
                    raise ValueError(
                        f"Cannot register {klass} as {name}. "
                        f"The name is already taken by {prev_processor}")
                DataProcessor.__task_dict__[name.lower()] = klass
            klass.task_name = names[0]
            return klass

        return decorator

    def get_train_examples(self) -> List[InputExample]:
        r"""Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError

    def get_dev_examples(self) -> List[InputExample]:
        r"""Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError

    def get_test_examples(self) -> List[InputExample]:
        r"""Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError

    @classmethod
    def _read_tsv(cls, input_file: Path,
                  quotechar: Optional[str] = None) -> List[List[str]]:
        """Reads a tab separated value file."""
        with input_file.open('r') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) == 0:
                    continue
                lines.append(line)
            return lines


def get_processor_class(task: str) -> Type[DataProcessor]:
    task = task.lower()
    klass = DataProcessor.__task_dict__.get(task, None)
    if klass is None:
        raise ValueError(f"Unsupported task {task}")
    return klass


class GLUEProcessor(DataProcessor, ABC):
    train_file = "train.tsv"
    dev_file = "dev.tsv"
    test_file = "test.tsv"
    label_column: int
    text_a_column: int
    text_b_column: int
    contains_header = True
    test_text_a_column: int
    test_text_b_column: int
    test_contains_header = True

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        if not hasattr(self, 'test_text_a_column'):
            self.test_text_a_column = self.text_a_column
        if not hasattr(self, 'test_text_b_column'):
            self.test_text_b_column = self.text_b_column

    def get_train_examples(self) -> List[InputExample]:
        return self._create_examples(
            self._read_tsv(self.data_dir / self.train_file), "train")

    def get_dev_examples(self) -> List[InputExample]:
        return self._create_examples(
            self._read_tsv(self.data_dir / self.dev_file), "dev")

    def get_test_examples(self) -> List[InputExample]:
        return self._create_examples(
            self._read_tsv(self.data_dir / self.test_file), "test")

    def _create_examples(self, lines: List[List[str]],
                         set_type: str) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and self.contains_header and set_type != "test":
                continue
            if i == 0 and self.test_contains_header and set_type == "test":
                continue
            guid = f"{set_type}-{i}"

            a_column = (self.text_a_column if set_type != "test" else
                        self.test_text_a_column)
            b_column = (self.text_b_column if set_type != "test" else
                        self.test_text_b_column)

            # there are some incomplete lines in QNLI
            if len(line) <= a_column:
                logging.warning('Incomplete line, ignored.')
                continue
            text_a = line[a_column]

            if b_column is not None:
                if len(line) <= b_column:
                    logging.warning('Incomplete line, ignored.')
                    continue
                text_b = line[b_column]
            else:
                text_b = None

            if set_type == "test":
                label = self.labels[0]
            else:
                if len(line) <= self.label_column:
                    logging.warning('Incomplete line, ignored.')
                    continue
                label = line[self.label_column]
            examples.append(InputExample(guid, text_a, text_b, label))
        return examples


@DataProcessor.register("MNLI", "MNLI_matched")
class MnliMatchedProcessor(GLUEProcessor):
    labels = ["contradiction", "entailment", "neutral"]

    dev_file = "dev_matched.tsv"
    test_file = "test_matched.tsv"
    label_column = -1
    text_a_column = 8
    text_b_column = 9


@DataProcessor.register("MNLI_mismatched")
class MnliMismatchedProcessor(MnliMatchedProcessor):
    dev_file = "dev_mismatched.tsv"
    test_file = "test_mismatched.tsv"


@DataProcessor.register("STS-B", "stsb")
class StsbProcessor(GLUEProcessor):
    labels: List[str] = []
    is_regression = True

    label_column = 9
    text_a_column = 7
    text_b_column = 8

    def _create_examples(self, lines: List[List[str]],
                         set_type: str) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and self.contains_header and set_type != "test":
                continue
            if i == 0 and self.test_contains_header and set_type == "test":
                continue
            guid = f"{set_type}-{i}"

            a_column = (self.text_a_column if set_type != "test" else
                        self.test_text_a_column)
            b_column = (self.text_b_column if set_type != "test" else
                        self.test_text_b_column)

            # there are some incomplete lines in QNLI
            if len(line) <= a_column:
                logging.warning('Incomplete line, ignored.')
                continue
            text_a = line[a_column]

            if b_column is not None:
                if len(line) <= b_column:
                    logging.warning('Incomplete line, ignored.')
                    continue
                text_b = line[b_column]
            else:
                text_b = None

            if set_type == "test":
                label = 0.0
            else:
                if len(line) <= self.label_column:
                    logging.warning('Incomplete line, ignored.')
                    continue
                label = float(line[self.label_column])
            examples.append(InputExample(guid, text_a, text_b, label))

        return examples


@DataProcessor.register("Yelp5")
class Yelp5Processor(DataProcessor):
    labels = ["1", "2", "3", "4", "5"]

    def get_train_examples(self) -> List[InputExample]:
        return self._create_examples(self.data_dir / "train.csv")

    def get_dev_examples(self) -> List[InputExample]:
        return self._create_examples(self.data_dir / "test.csv")

    def get_test_examples(self):
        raise TypeError("The Yelp 5 dataset does not have a test set.")

    @staticmethod
    def _create_examples(input_file: Path) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        with input_file.open() as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                label = line[0]
                text_a = line[1].replace('""', '"').replace('\\"', '"')
                examples.append(InputExample(
                    guid=str(i), text_a=text_a, text_b=None, label=label))
        return examples


@DataProcessor.register("IMDB")
class ImdbProcessor(DataProcessor):
    labels = ["neg", "pos"]

    def get_train_examples(self) -> List[InputExample]:
        return self._create_examples(self.data_dir / "train")

    def get_dev_examples(self) -> List[InputExample]:
        return self._create_examples(self.data_dir / "test")

    def get_test_examples(self):
        raise TypeError("The IMDB dataset does not have a test set.")

    @staticmethod
    def _create_examples(data_dir: Path) -> List[InputExample]:
        examples = []
        for label in ["neg", "pos"]:
            cur_dir = data_dir / label
            for filename in cur_dir.iterdir():
                if filename.suffix != ".txt":
                    continue
                with filename.open() as f:
                    text = f.read().strip().replace("<br />", " ")
                examples.append(InputExample(
                    guid=str(filename), text_a=text, text_b=None, label=label))
        return examples
