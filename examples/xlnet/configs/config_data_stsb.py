# Random seed. Set to `None` to disable.
seed = 19260817
# Name of the task to train on.
task = "STS-B"

# Maximum sequence length.
max_seq_len = 128
# Batch size during training.
batch_size = 4
# Accumulate gradients across multiple batches before performing an
# optimizer step.
backwards_per_step = 8
# Batch size during evaluation.
eval_batch_size = 64

# Number of steps to train.
train_steps = 1200
# Base (maximum) learning rate.
lr = 5e-5
# Number of warm-up steps. Learning rate linearly grows to base LR from zero in
# this many steps.
warmup_steps = 120
# Per-layer LR scaling coefficient.
# - Topmost (highest ID) layer: LR[N] = lr.
# - Lower layers: LR[x - 1] = LR[x] * lr_layer_decay_rate.
lr_layer_decay_rate = 1.0
# Ratio of minimum LR. By end of training, LR will become lr * min_lr_ratio.
min_lr_ratio = 0.0
# Epsilon value for Adam optimizer.
adam_eps = 1e-8
# Weight decay rate. When value is greater than zero, BertAdam optimizer
# is used.
weight_decay = 0.0
# Maximum norm for gradient clipping.
grad_clip = 1.0

# Display training stats per this many steps.
display_steps = 100
# Evaluate model per this many steps. Set to -1 to only evaluate
# at end of training.
eval_steps = 500
# Save model per this many steps. Set to -1 to only evaluate
# at end of training.
save_steps = 500
