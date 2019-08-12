# Copyright 2018 The Texar Authors. All Rights Reserved.
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

dataset = "yahoo"
num_epochs = 100
hidden_size = 512
dec_dropout_in = 0.
enc_dropout_in = 0.
enc_dropout_out = 0.
batch_size = 32
embed_dim = 512

latent_dims = 32

lr_decay_hparams = {
    "init_lr": 0.001,
    "threshold": 2,
    "decay_factor": 0.5,
    "max_decay": 5
}


relu_dropout = 0.2
embedding_dropout = 0.2
attention_dropout = 0.2
residual_dropout = 0.2
num_blocks = 3

decoder_type = 'transformer'

enc_cell_hparams = {
    "type": "LSTMCell",
    "kwargs": {
        "num_units": hidden_size,
        "bias": 0.
    },
    "dropout": {"output_keep_prob": 1. - enc_dropout_out},
    "num_layers": 1
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


max_pos = 300    # max sequence length in training data
dec_pos_emb_hparams = {
    'dim': hidden_size,
}

# due to the residual connection, the embed_dim should be equal to hidden_size
trans_hparams = {
    'output_layer_bias': False,
    'embedding_dropout': embedding_dropout,
    'residual_dropout': residual_dropout,
    'num_blocks': num_blocks,
    'dim': hidden_size,
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'factor': 1.0,
            'mode': 'FAN_AVG',
            'uniform': True,
        },
    },
    'multihead_attention': {
        'dropout_rate': attention_dropout,
        'num_heads': 8,
        'num_units': hidden_size,
        'output_dim': hidden_size
    },
    'poswise_feedforward': {
        'name': 'fnn',
       'layers': [
            {
                'type': 'Linear',
                'kwargs': {
                    "in_features": hidden_size,
                    "out_features": hidden_size * 4,
                    "bias": True,
                },
            },
            {
                'type': 'ReLU',
            },
            {
                'type': 'Dropout',
                'kwargs': {
                    'p': relu_dropout,
                }
            },
            {
                'type': 'Linear',
                'kwargs': {
                    "in_features": hidden_size * 4,
                    "out_features": hidden_size,
                    "bias": True,
                }
            }
        ],
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
        "files": './data/yahoo/yahoo.train.txt',
        "vocab_file": './data/yahoo/vocab.txt'
    }
}

val_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": './data/yahoo/yahoo.valid.txt',
        "vocab_file": './data/yahoo/vocab.txt'
    }
}

test_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "dataset": {
        "files": './data/yahoo/yahoo.test.txt',
        "vocab_file": './data/yahoo/vocab.txt'
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
