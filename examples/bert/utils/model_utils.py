"""
Model utility functions
"""
import json
import collections
import re
import sys
import random
import tensorflow as tf
import torch


def transform_bert_to_texar_config(input_json):
    """
    Load the Json config file and transform it into Texar style configuration.
    """
    config_ckpt = json.loads(
        open(input_json).read())

    configs = {}
    configs['hidden_size'] = config_ckpt['hidden_size']
    hidden_dim = config_ckpt['hidden_size']
    configs['embed'] = {
        'name': 'word_embeddings',
        'dim': hidden_dim}
    configs['vocab_size'] = config_ckpt['vocab_size']

    configs['segment_embed'] = {
        'name': 'token_type_embeddings',
        'dim': hidden_dim}
    configs['type_vocab_size'] = config_ckpt['type_vocab_size']

    configs['position_embed'] = {
        'name': 'position_embeddings',
        'dim': hidden_dim}
    configs['position_size'] = config_ckpt['max_position_embeddings']

    configs['encoder'] = {
        'name': 'encoder',
        'embedding_dropout': config_ckpt['hidden_dropout_prob'],
        'num_blocks': config_ckpt['num_hidden_layers'],
        'multihead_attention': {
            'use_bias': True,
            'num_units': hidden_dim,
            'num_heads': config_ckpt['num_attention_heads'],
            'output_dim': hidden_dim,
            'dropout_rate': config_ckpt['attention_probs_dropout_prob'],
            'name': 'self'
        },
        'residual_dropout': config_ckpt['hidden_dropout_prob'],
        'dim': hidden_dim,
        'use_bert_config': True,
        'poswise_feedforward': {
            "layers": [
                {
                    "type": "Linear",
                    "kwargs": {
                        "in_features": hidden_dim,
                        "out_features": config_ckpt['intermediate_size'],
                        "bias": True,
                    },
                },
                {
                    'type': config_ckpt['hidden_act'].upper()
                },
                {
                    "type": "Linear",
                    "kwargs": {
                        "in_features": config_ckpt['intermediate_size'],
                        "out_features": hidden_dim,
                        "bias": True,
                    },
                },
            ],
        },
    }
    return configs


def get_lr(global_step: int, num_train_steps, num_warmup_steps, static_lr,
           end_learning_rate=1e-4):
    """
    Calculate the learinng rate given global step and warmup steps.
    The learinng rate is following a linear warmup and linear decay.
    """
    _lr = static_lr
    global_step = min(global_step, num_warmup_steps)

    decayed_learning_rate = (_lr - end_learning_rate) *\
        (1 - global_step / num_train_steps) ** 1.0 + end_learning_rate

    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = float(global_step)
        warmup_steps_float = float(warmup_steps_int)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = static_lr * warmup_percent_done

        is_warmup = float(global_steps_int < warmup_steps_int)
        learning_rate = ((1.0 - is_warmup) * decayed_learning_rate
                         + is_warmup * warmup_learning_rate)

    return learning_rate


def init_bert_checkpoint(model, init_checkpoint):
    """
    Initializes BERT model parameters from a checkpoint provided by
    Google.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
        import os
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to "
              "be installed. Please see https://www.tensorflow.org/install/ "
              "for installation instructions.")
        raise

    global_tensor_map = {
        'bert/embeddings/word_embeddings': 'word_embedder._embedding',
        'bert/embeddings/token_type_embeddings':
            'segment_embedder._embedding',
        'bert/embeddings/position_embeddings':
            'position_embedder._embedding',
        'bert/embeddings/LayerNorm/beta':
            'encoder.input_normalizer.bias',
        'bert/embeddings/LayerNorm/gamma':
            'encoder.input_normalizer.weight',
    }
    layer_tensor_map = {
        "attention/self/key/bias": "self_attns.{}.K_dense.bias",
        "attention/self/query/bias": "self_attns.{}.Q_dense.bias",
        "attention/self/value/bias": "self_attns.{}.V_dense.bias",
        "attention/output/dense/bias": "self_attns.{}.O_dense.bias",
        "attention/output/LayerNorm/gamma": "poswise_layer_norm.{}.weight",
        "attention/output/LayerNorm/beta": "poswise_layer_norm.{}.bias",
        "intermediate/dense/bias": "poswise_networks.{}._layers.0.bias",
        "output/dense/bias": "poswise_networks.{}._layers.2.bias",
        "output/LayerNorm/gamma": "output_layer_norm.{}.weight",
        "output/LayerNorm/beta": "output_layer_norm.{}.bias",
    }
    layer_transpose_map = {
        "attention/self/key/kernel": "self_attns.{}.K_dense.weight",
        "attention/self/query/kernel": "self_attns.{}.Q_dense.weight",
        "attention/self/value/kernel": "self_attns.{}.V_dense.weight",
        "attention/output/dense/kernel": "self_attns.{}.O_dense.weight",
        "intermediate/dense/kernel": "poswise_networks.{}._layers.0.weight",
        "output/dense/kernel": "poswise_networks.{}._layers.2.weight",
    }
    pooler_map = {
        'bert/pooler/dense/bias': 'pooler.0.bias',
        'bert/pooler/dense/kernel': 'pooler.0.weight'
    }
    tf_path = os.path.abspath(init_checkpoint)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tfnames, arrays = [], []
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        tfnames.append(name)
        arrays.append(array.squeeze())

    py_prefix = "encoder."

    idx = 0
    for name, array in zip(tfnames, arrays):
        processing = (idx + 1.0) / len(tfnames)

        if not name.startswith('bert'):
            continue

        if name in global_tensor_map:
            v_name = global_tensor_map[name]
            pointer = name_to_variable(model, v_name)
            assert pointer.shape == array.shape
            pointer.data = torch.from_numpy(array)
            idx += 1
        elif name in pooler_map:
            pointer = name_to_variable(model, pooler_map[name])
            if name.endswith('bias'):
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)
                idx += 1
            else:
                array_t = np.transpose(array)
                assert pointer.shape == array_t.shape
                pointer.data = torch.from_numpy(array_t)
                idx += 1
        else:
            # here name is the tensorflow variable name
            name_tmp = name.split("/")
            # e.g. layer_
            layer_no = name_tmp[2][6:]
            print('layer_no:{}'.format(layer_no))
            name_tmp = "/".join(name_tmp[3:])
            if name_tmp in layer_tensor_map:
                v_name = layer_tensor_map[name_tmp].format(layer_no)
                pointer = name_to_variable(model, py_prefix + v_name)
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)
            elif name_tmp in layer_transpose_map:
                v_name = layer_transpose_map[name_tmp].format(layer_no)
                pointer = name_to_variable(model, py_prefix + v_name)
                array_t = np.transpose(array)
                assert pointer.shape == array_t.shape
                pointer.data = torch.from_numpy(array_t)
            else:
                print("Name error", name)
                raise Exception
            idx += 1
    print('initialized {} variables'.format(idx))


def name_to_variable(model, name):
    """
    Find the corresponding varible give the specified name
    :param model:
    :param name:
    :return:
    """
    pointer = model
    name = name.split(".")
    for m_name in name:
        if m_name.isdigit():
            num = int(m_name)
            pointer = pointer[num]
        else:
            pointer = getattr(pointer, m_name)
    return pointer
