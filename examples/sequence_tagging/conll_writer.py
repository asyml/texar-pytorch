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
Writer module.
"""


class CoNLLWriter:

    def __init__(self, i2w, i2n):
        self.__source_file = None
        self.__i2w = i2w
        self.__i2n = i2n

    def start(self, file_path):
        self.__source_file = open(file_path, 'w', encoding='utf-8')

    def close(self):
        self.__source_file.close()

    def write(self, word, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__i2w[word[i, j]]
                tgt = self.__i2n[targets[i, j]]
                pred = self.__i2n[predictions[i, j]]
                self.__source_file.write(
                    '%d %s %s %s %s %s\n' % (j + 1, w, "_", "_", tgt, pred))
            self.__source_file.write('\n')
