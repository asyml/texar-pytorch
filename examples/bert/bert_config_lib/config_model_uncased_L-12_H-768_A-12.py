embed = {"dim": 768, "name": "word_embeddings"}
vocab_size = 30522

segment_embed = {"dim": 768, "name": "token_type_embeddings"}
type_vocab_size = 2

position_embed = {"dim": 768, "name": "position_embeddings"}
position_size = 512

encoder = {
    "dim": 768,
    "embedding_dropout": 0.1,
    "multihead_attention": {
        "dropout_rate": 0.1,
        "name": "self",
        "num_heads": 12,
        "num_units": 768,
        "output_dim": 768,
        "use_bias": True,
    },
    "name": "encoder",
    "num_blocks": 12,
    "poswise_feedforward": {
        "layers": [
            {
                "kwargs": {
                    "bias": True,
                    "in_features": 768,
                    "out_features": 3072,
                },
                "type": "Linear",
            },
            {
                "type": "GELU",
            },
            {
                "kwargs": {
                    "bias": True,
                    "in_features": 3072,
                    "out_features": 768,
                },
                "type": "Linear",
            },
        ]
    },
    "residual_dropout": 0.1,
    "use_bert_config": True,
}

output_size = 768  # The output dimension of BERT

