"""Texar config file of the GPT-2 model_117M model.
"""

vocab_size = 50257
dim = 768

embed = {
    "dim": dim,
}

pos_embed = {
    'dim': dim
}
position_size = 1024

decoder = {
    "dim": dim,
    "num_blocks": 12,
    "use_gpt_config": True,
    'embedding_dropout': 0.0,
    'residual_dropout': 0.0,
    "multihead_attention": {
        "use_bias": True,
        "num_units": dim,
        "num_heads": 12,
        "dropout_rate": 0.0,
        "output_dim": dim,
    },
    "initializer": {
        "type": "variance_scaling_initializer",
        "kwargs": {
            "factor": 1.0,
            "mode": "FAN_AVG",
            "uniform": True,
        },
    },
    "poswise_feedforward": {
        "layers": [
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": dim,
                    "out_features": dim * 4,
                    "bias": True,
                }
            },
            {
                "type": "GPTGELU",
                "kwargs": {
                }
            },
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": dim * 4,
                    "out_features": dim,
                    "bias": True,
                }
            }
        ],
        "name": "ffn",
    },
}
