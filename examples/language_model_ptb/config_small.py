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
"""PTB LM small size config.
"""

init_scale = 0.1
num_epochs = 13
hidden_size = 200
keep_prob = 1.0
batch_size = 20
num_steps = 20

cell = {
    "type": "LSTMCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {
        "output_keep_prob": keep_prob
    },
    "num_layers": 2
}

emb = {
    "dim": hidden_size,
    "dropout_rate": 1 - keep_prob,
    "initializer": {
        "type": "torch.nn.init.uniform_",
        "kwargs": {
            "a": -init_scale,
            "b": init_scale,
        }
    },
}

opt = {
    "optimizer": {
        "type": "SGD",
        "kwargs": {
            "lr": 1.0
        }
    },
    "gradient_clip": {
        "type": "clip_grad_norm_",
        "kwargs": {
            "max_norm": 5.
        }
    },
    "learning_rate_decay": {
        "type": "ExponentialLR",
        "kwargs": {
            "gamma": 0.5,
        },
    }
}
