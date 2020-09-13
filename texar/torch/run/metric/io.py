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
Executor metrics that are actually not metrics, but for IO.
"""
from pathlib import Path
from typing import TypeVar, Union

from texar.torch.run.metric.base_metric import SimpleMetric

__all__ = [
    "FileWriterMetric",
]

Input = TypeVar('Input')
Value = TypeVar('Value')


class FileWriterMetric(SimpleMetric[Input, None]):
    r"""A metric for writing predictions to file.

    Args:
        file_path: Path to the output file. You can include :class:`Executor`
            status variables in the path using Python's string formatting
            syntax. For example, ``output_{split}_{epoch}.txt`` will resolve to
            ``output_test_1.txt`` if the metric is used in testing after
            epoch 1. Available variables are: ``epoch``, ``iteration``,
            and ``split``.
        mode (str): Mode to open the file in. Defaults to ``"w"`` (write mode).
        sep (str): Separator for different examples. Defaults to new line
            (``"\n"``).

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
    """
    requires_pred = True
    requires_label = False

    def __init__(self, file_path: Union[str, Path], mode: str = "w",
                 sep: str = "\n", *, pred_name: str):
        super().__init__(pred_name=pred_name, label_name=None)
        self.file_path = file_path
        self.mode = mode
        self.sep = sep

    def _value(self) -> None:
        pass

    def finalize(self, executor) -> None:
        path = str(self.file_path)
        for key in ["epoch", "iteration", "split"]:
            if "{" + key + "}" in path:
                path = path.replace("{" + key + "}", str(executor.status[key]))
        with open(path, self.mode) as writer:
            writer.write(self.sep.join(str(p) for p in self.predicted))
