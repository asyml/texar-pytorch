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
Data processor base class. Adapted from
https://github.com/zihangdai/xlnet/blob/master/run_classifier.py#L144-L192
"""

import csv
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
