"""Config file for GPT2 training.
"""

pickle_data_dir = "data/toy"
max_seq_length = 128
max_decoding_length = max_seq_length

train_batch_size = 32
max_train_epoch = 100
display_steps = 1  # Print training loss every display_steps; -1 to disable
eval_steps = 1  # Eval on the dev set every eval_steps; -1 to disable

eval_batch_size = 8
test_batch_size = 8

# Optimization configs

opt = {
    'optimizer': {
        'type': 'Adam',
        'kwargs': {
            'lr': 0.001
        }
    }
}

# Data configs

feature_types = {
    # Reading features from pickle data file.
    # E.g., Reading feature "text_ids" as dtype `int64`;
    # "stacked_tensor" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "text_ids": ["int64", "stacked_tensor", max_seq_length],
    "length": ["int64", "stacked_tensor"]
}

train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": feature_types,
        "files": "{}/train.pkl".format(pickle_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": 10000
}

eval_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": feature_types,
        "files": "{}/dev.pkl".format(pickle_data_dir)
    },
    "shuffle": False
}

# Set to `test_hparam` to `None` if generating from scratch
# (instead of generating continuation) at test time
test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": feature_types,
        "files": "{}/test.pkl".format(pickle_data_dir)
    },
    "shuffle": False
}
