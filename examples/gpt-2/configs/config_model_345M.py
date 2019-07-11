"""Texar config file of the GPT-2 model_345M model.
"""
pretrained_model_name = "345M"

vocab_size = 50257

embed = {
    "dim": 1024,
}

pos_embed = {
    "dim": 1024
}
position_size = 1024

decoder = {
    "dim": 1024,
    "num_blocks": 24,
    "use_gpt_config": True,
    "embedding_dropout": 0.0,
    "residual_dropout": 0.0,
    "multihead_attention": {
        "use_bias": True,
        "num_units": 1024,
        "num_heads": 16,
        "dropout_rate": 0.0,
        "output_dim": 1024,
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
                    "in_features": 1024,
                    "out_features": 1024 * 4,
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
                    "in_features": 1024 * 4,
                    "out_features": 1024,
                    "bias": True,
                }
            }
        ],
        "name": "ffn",
    },
}
