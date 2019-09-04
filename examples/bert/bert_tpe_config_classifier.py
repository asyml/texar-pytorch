name = "bert_classifier"
hidden_size = 768
clas_strategy = "cls_time"
dropout = 0.1
num_classes = 2

# hyperparams
hyperparams = {
    "optimizer.warmup_steps": {"start": 10000, "end": 20000, "dtype": int},
    "optimizer.static_lr": {"start": 1e-3, "end": 1e-2, "dtype": float}
}
