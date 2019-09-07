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

# type: ignore
"""Sentence convolutional classifier config.

This is (approximately) the config of the paper:
(Kim) Convolutional Neural Networks for Sentence Classification
  https://arxiv.org/pdf/1408.5882.pdf
"""

import copy

num_epochs = 15
max_seq_length = 56
emb_dim = 300

train_data = {
    "batch_size": 50,
    "datasets": [
        {
            "files": "./data/sst2.train.sentences.txt",
            "vocab_file": "./data/sst2.vocab",
            # Discards samples with length > 56
            "max_seq_length": max_seq_length,
            "length_filter_mode": "discard",
            # Do not append BOS/EOS tokens to the sentences
            "bos_token": "",
            "eos_token": "",
            "data_name": "sentence"
        },
        {
            "files": "./data/sst2.train.labels.txt",
            "data_type": "int64",
            "data_name": "label"
        }
    ]
}
# The val and test data have the same config with the train data, except
# for the file names
val_data = copy.deepcopy(train_data)
val_data["datasets"][0]["files"] = "./data/sst2.dev.sentences.txt"
val_data["datasets"][1]["files"] = "./data/sst2.dev.labels.txt"
test_data = copy.deepcopy(train_data)
test_data["datasets"][0]["files"] = "./data/sst2.test.sentences.txt"
test_data["datasets"][1]["files"] = "./data/sst2.test.labels.txt"

# Word embedding
emb = {
    "dim": emb_dim
}

# Classifier
clas = {
    "num_conv_layers": 1,
    "out_channels": 100,
    "kernel_size": [3, 4, 5],
    "conv_activation": "ReLU",
    "pooling": "MaxPool1d",
    "num_dense_layers": 0,
    "dropout_conv": [1],
    "dropout_rate": 0.5,
    "num_classes": 2
}

# Optimization
# Just use the default config, e.g., Adam Optimizer
opt = {}
