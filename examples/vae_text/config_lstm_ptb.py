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
"""VAE config.
"""

dataset = "ptb"
num_epochs = 100
hidden_size = 256
dec_dropout_in = 0.5
dec_dropout_out = 0.5
enc_dropout_in = 0.
enc_dropout_out = 0.
word_keep_prob = 0.5
batch_size = 32
embed_dim = 256

latent_dims = 32

lr_decay_hparams = {
    "init_lr": 0.001,
    "threshold": 2,
    "decay_factor": 0.5,
    "max_decay": 5
}


decoder_type = 'lstm'

enc_cell_hparams = {
    "type": "LSTMCell",
    "kwargs": {
        "num_units": hidden_size,
        "bias": 0.
    },
    "dropout": {"output_keep_prob": 1. - enc_dropout_out},
    "num_layers": 1
}

dec_cell_hparams = {
    "type": "LSTMCell",
    "kwargs": {
        "num_units": hidden_size,
        "bias": 0.,
    },
    "dropout": {"output_keep_prob": 1. - dec_dropout_out},
    "num_layers": 1,
}

enc_emb_hparams = {
    'name': 'lookup_table',
    "dim": embed_dim,
    "dropout_rate": enc_dropout_in,
    'initializer': {
        'type': 'normal_',
        'kwargs': {
            'mean': 0.0,
            'std': embed_dim**-0.5,
        },
    }
}

dec_emb_hparams = {
    'name': 'lookup_table',
    "dim": embed_dim,
    "dropout_rate": dec_dropout_in,
    'initializer': {
        'type': 'normal_',
        'kwargs': {
            'mean': 0.0,
            'std': embed_dim**-0.5,
        },
    }
}

# KL annealing
kl_anneal_hparams = {
    "warm_up": 10,
    "start": 0.1
}

train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": './simple-examples/data/ptb.train.txt',
        "vocab_file": './simple-examples/data/vocab.txt'
    }
}

val_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": './simple-examples/data/ptb.valid.txt',
        "vocab_file": './simple-examples/data/vocab.txt'
    }
}

test_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "dataset": {
        "files": './simple-examples/data/ptb.test.txt',
        "vocab_file": './simple-examples/data/vocab.txt'
    }
}

opt_hparams = {
    'optimizer': {
        'type': 'Adam',
        'kwargs': {
            'lr': 0.001
        }
    },
    'gradient_clip': {
        "type": "clip_grad_norm_",
        "kwargs": {
            "max_norm": 5,
            "norm_type": 2
        }
    }
}
