# The full possible hyperparameters for the attentional seq2seq model.
# Most of the hyperparameters take the default values and are not necessary to
# specify explicitly. The config here results in the same model with the
# `config_model.py`.

num_units = 256
beam_width = 10

# --------------------- Embedder --------------------- #
embedder = {
    'dim': num_units,
    "initializer": None,
    "dropout_rate": 0.,
    "dropout_strategy": 'element',
    "name": "word_embedder",
}

# --------------------- Encoder --------------------- #
encoder = {
    'rnn_cell_fw': {
        'type': 'LSTMCell',
        'kwargs': {
            'num_units': num_units,
        },
        'num_layers': 1,
        'dropout': {
            'input_keep_prob': 1.0,
            'output_keep_prob': 1.0,
            'state_keep_prob': 1.0,
            'variational_recurrent': False,
        },
        'residual': False,
        'highway': False,
    },
    'rnn_cell_bw': {
        'type': 'LSTMCell',
        'kwargs': {
            'num_units': num_units,
        },
        'num_layers': 1,
        'dropout': {
            'input_keep_prob': 1.0,
            'output_keep_prob': 1.0,
            'state_keep_prob': 1.0,
            'variational_recurrent': False,
        },
        'residual': False,
        'highway': False,
    },
    'rnn_cell_share_config': True,
    'output_layer_fw': {
        "num_layers": 0,
        "layer_size": 128,
        "activation": "Identity",
        "final_layer_activation": None,
        "other_dense_kwargs": None,
        "dropout_layer_ids": [],
        "dropout_rate": 0.5,
        "variational_dropout": False,
    },
    'output_layer_bw': {
        "num_layers": 0,
        "layer_size": 128,
        "activation": "Identity",
        "final_layer_activation": None,
        "other_dense_kwargs": None,
        "dropout_layer_ids": [],
        "dropout_rate": 0.5,
        "variational_dropout": False,
    },
    'output_layer_share_config': True,
    'name': 'bidirectional_rnn_encoder'
}

# --------------------- Decoder --------------------- #
decoder = {
    'rnn_cell': {
        'type': 'LSTMCell',
        'kwargs': {
            'num_units': 256,
        },
        'num_layers': 1,
        'dropout': {
            'input_keep_prob': 1.0,
            'output_keep_prob': 1.0,
            'state_keep_prob': 1.0,
            'variational_recurrent': False,
        },
        'residual': False,
        'highway': False,
    },
    'attention': {
        "type": "LuongAttention",
        "kwargs": {
            "num_units": 256,
        },
        "attention_layer_size": 256,
        "alignment_history": False,
        "output_attention": True,
    },
    'helper_train': {
        'type': 'TrainingHelper',
        'kwargs': {}
    },
    'helper_infer': {
        'type': 'SampleEmbeddingHelper',
        'kwargs': {}
    },
    'max_decoding_length_train': None,
    'max_decoding_length_infer': 60,
    'output_layer_bias': True,
    'name': 'attention_rnn_decoder'
}
# --------------------- Optimization --------------------- #
opt = {
    'optimizer': {
        'type':  'Adam',
        'kwargs': {
            'lr': 0.001,
        },
    },
    'learning_rate_decay': {
            "type": "",
            "kwargs": {}
        },
    'gradient_clip': {
            "type": "",
            "kwargs": {}
        },
    'gradient_noise_scale': None,
    'name': None
}
