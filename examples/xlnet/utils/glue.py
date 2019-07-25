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
Data processors for GLUE datasets. Adapted from
https://github.com/zihangdai/xlnet/blob/master/run_classifier.py
"""

import logging
from abc import ABC
from typing import List

from utils.processor import InputExample, DataProcessor


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
