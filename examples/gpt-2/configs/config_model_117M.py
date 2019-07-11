"""Texar config file of the GPT-2 model_117M model.
"""
pretrained_model_name = "117M"

vocab_size = 50257

embed = {
    "dim": 768,
}

position_embed = {
    'dim': 768
}
position_size = 1024

decoder = {
    "dim": 768,
    "num_blocks": 12,
    "use_gpt_config": True,
    'embedding_dropout': 0.0,
    'residual_dropout': 0.0,
    "multihead_attention": {
        "use_bias": True,
        "num_units": 768,
        "num_heads": 12,
        "dropout_rate": 0.0,
        "output_dim": 768,
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
                    "in_features": 768,
                    "out_features": 768 * 4,
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
                    "in_features": 768 * 4,
                    "out_features": 768,
                    "bias": True,
                }
            }
        ],
        "name": "ffn",
    },
}
