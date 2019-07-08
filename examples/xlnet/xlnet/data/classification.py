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
Data processors for classification datasets. Adapted from
https://github.com/zihangdai/xlnet/blob/master/run_classifier.py
"""

import csv
import os
from typing import List

from xlnet.data.processor import DataProcessor, InputExample


@DataProcessor.register("Yelp5")
class Yelp5Processor(DataProcessor):
    labels = ["1", "2", "3", "4", "5"]

    def get_train_examples(self) -> List[InputExample]:
        return self._create_examples(self.data_dir / "train.csv")

    def get_dev_examples(self) -> List[InputExample]:
        return self._create_examples(self.data_dir / "test.csv")

    def get_test_examples(self):  # pylint: disable=no-self-use
        raise TypeError("The Yelp 5 dataset does not have a test set.")

    @staticmethod
    def _create_examples(input_file: str) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        with open(input_file) as f:
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

    def get_test_examples(self):  # pylint: disable=no-self-use
        raise TypeError("The IMDB dataset does not have a test set.")

    @staticmethod
    def _create_examples(data_dir: str) -> List[InputExample]:
        examples = []
        for label in ["neg", "pos"]:
            cur_dir = os.path.join(data_dir, label)
            for filename in os.listdir(cur_dir):
                if not filename.endswith("txt"):
                    continue
                path = os.path.join(cur_dir, filename)
                with open(path) as f:
                    text = f.read().strip().replace("<br />", " ")
                examples.append(InputExample(
                    guid=filename, text_a=text, text_b=None, label=label))
        return examples
