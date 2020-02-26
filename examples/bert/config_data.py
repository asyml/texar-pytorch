pickle_data_dir = "data/MRPC"
max_seq_length = 128
num_classes = 2
num_train_data = 3668

# used for bert executor example
max_batch_tokens = 128

train_batch_size = 32
max_train_epoch = 5
display_steps = 50  # Print training loss every display_steps; -1 to disable

# tbx config
tbx_logging_steps = 5  # log the metrics for tbX visualization
tbx_log_dir = "runs/"
exp_number = 1  # experiment number

eval_steps = 100  # Eval on the dev set every eval_steps; -1 to disable
# Proportion of training to perform linear learning rate warmup for.
# E.g., 0.1 = 10% of training.
warmup_proportion = 0.1
eval_batch_size = 8
test_batch_size = 8

feature_types = {
    # Reading features from pickled data file.
    # E.g., Reading feature "input_ids" as dtype `int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "input_ids": ["int64", "stacked_tensor", max_seq_length],
    "input_mask": ["int64", "stacked_tensor", max_seq_length],
    "segment_ids": ["int64", "stacked_tensor", max_seq_length],
    "label_ids": ["int64", "stacked_tensor"]
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
    "shuffle_buffer_size": 100
}

eval_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": feature_types,
        "files": "{}/eval.pkl".format(pickle_data_dir)
    },
    "shuffle": False
}

test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": feature_types,
        "files": "{}/predict.pkl".format(pickle_data_dir)
    },
    "shuffle": False
}
