# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Various optimization related utilities.
"""

from typing import Dict, Any, List, Optional, Union, Callable

import torch
from torch.optim.optimizer import Optimizer
from  torch.optim.lr_scheduler import _LRScheduler

from texar.hyperparams import HParams
from texar.utils import utils

# pylint: disable=too-many-arguments, no-member

__all__ = [
    "default_optimization_hparams",
    "get_optimizer",
    "get_train_op"
]


def default_optimization_hparams() -> Dict[str, Any]:
    """Returns a `dict` of default hyperparameters of training op
    and their default values

    .. role:: python(code)
       :language: python

    .. code-block:: python

        {
            "optimizer": {
                "type": "Adam",
                "kwargs": {
                    "lr": 0.001
                }
            },
            "learning_rate_decay": {
                "type": "",
                "kwargs": {}
            },
            "gradient_clip": {
                "type": "",
                "kwargs": {}
            },
            "gradient_noise_scale": None,
            "name": None
        }

    Here:

    "optimizer" : dict
        Hyperparameters of a :tf_main:`tf.train.Optimizer <train/Optimizer>`.

        - **"type"** specifies the optimizer class. This can be

            - The string name or full module path of an optimizer class. \
            If the class name is provided, the class must be in module \
            :tf_main:`tf.train <train>`, \
            :tf_main:`tf.contrib.opt <contrib/opt>` or :mod:`texar.custom` \
            , :mod:`texar.core.optimization`
            - An optimizer class.
            - An instance of an optimizer class.

            For example

            .. code-block:: python

                "type": "AdamOptimizer" # class name
                "type": "my_module.MyOptimizer" # module path
                "type": tf.contrib.opt.AdamWOptimizer # class
                "type": my_module.MyOptimizer # class
                "type": GradientDescentOptimizer(learning_rate=0.1) # instance
                "type": MyOptimizer(...) # instance

        - **"kwargs"** is a `dict` specifying keyword arguments for creating \
        the optimizer class instance, with :python:`opt_class(**kwargs)`. \
        Ignored if "type" is a class instance.

    "learning_rate_decay" : dict
        Hyperparameters of learning rate decay function. The learning rate
        starts decay from :attr:`"start_decay_step"` and keeps unchanged after
        :attr:`"end_decay_step"` or reaching :attr:`"min_learning_rate"`.

        The decay function is specified in "type" and "kwargs".

            - "type" can be a decay function or its name or module path. If \
            function name is provided, it must be from module \
            :tf_main:`tf.train <train>` or :mod:`texar.custom`, \
            :mod:`texar.core.optimization`.

            - "kwargs" is a `dict` of keyword arguments for the function \
            excluding arguments named "global_step" and "learning_rate".

        The function is called with
        :python:`lr = decay_fn(learning_rate=lr, global_step=offset_step,
        **kwargs)`, where `offset_step` is the global step offset as above.
        The only exception is :tf_main:`tf.train.piecewise_constant
        <train/piecewise_constant>` which is called with
        :python:`lr = piecewise_constant(x=offset_step, **kwargs)`.

    "gradient_clip" : dict
        Hyperparameters of gradient clipping. The gradient clipping function
        takes a list of `(gradients, variables)` tuples and returns a list
        of `(clipped_gradients, variables)` tuples. Typical examples include
        :tf_main:`tf.clip_by_global_norm <clip_by_global_norm>`,
        :tf_main:`tf.clip_by_value <clip_by_value>`,
        :tf_main:`tf.clip_by_norm <clip_by_norm>`,
        :tf_main:`tf.clip_by_average_norm <clip_by_average_norm>`, etc.

        "type" specifies the gradient clip function, and can be a function,
        or its name or mudule path. If function name is provided, the
        function must be from module :tf_main:`tf < >` or :mod:`texar.custom`,
        :mod:`texar.core.optimization`.


        "kwargs" specifies keyword arguments to the function, except arguments
        named "t" or "t_list".

        The function is called with
        :python:`clipped_grads(, _) = clip_fn(t_list=grads, **kwargs)`
        (e.g., for :tf_main:`tf.clip_by_global_norm <clip_by_global_norm>`) or
        :python:`clipped_grads = [clip_fn(t=grad, **kwargs) for grad in grads]`
        (e.g., for :tf_main:`tf.clip_by_value <clip_by_value>`).

    "gradient_noise_scale" : float, optional
        Adds 0-mean normal noise scaled by this value to gradient.
    """
    return {
        "optimizer": {
            "type": "Adam",
            "kwargs": {
                "lr": 0.001
            }
        },
        "learning_rate_decay": {
            "type": "",
            "kwargs": {}
        },
        "gradient_clip": {
            "type": "",
            "kwargs": {}
        },
        "gradient_noise_scale": None,
        # TODO(zhiting): allow module-level control of gradient_multipliers
        "name": None
    }


def get_optimizer(
        params: Union[List[torch.Tensor], List[Dict[str, List[torch.Tensor]]]],
        hparams: Optional[Union[HParams, Dict[str, Any]]] = None) -> \
        Optimizer:
    """Creates a optimizer instance.

        Args:
            params: an iterable of :class:`torch.Tensor` s or
                :class:`dict` s. Specifies what Tensors should be optimized.
            hparams (dict or HParams, optional): hyperparameters. Missing
                hyperparameters are set to default values automatically. See
                :func:`~texar.core.default_optimization_hparams` for
                all hyperparameters and default values.

        Returns:
            optimizer: the torch.optim.Optimizer instance specified in hparams.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_optimization_hparams())

    hparams_opt = hparams["optimizer"]

    optimizer_type = hparams_opt["type"]
    if isinstance(optimizer_type, Optimizer):
        optimizer_class = optimizer_type
    else:
        optimizer_modules = ['torch.optim',
                             'texar.custom']
        try:
            optimizer_class = utils.check_or_get_class(  # type: ignore
                optimizer_type, optimizer_modules, Optimizer)
        except TypeError:
            raise ValueError(
                "Unrecognized optimizer. Must be string name of the "
                "optimizer class, or the class which is a subclass of "
                "torch.optim.Optimizer, or an instance of the subclass of "
                "Optimizer.")

    optimizer_kwargs = hparams_opt["kwargs"].todict()
    optimizer_kwargs.update({"params": params})
    optimizer = optimizer_class(**optimizer_kwargs)  # type: ignore

    return optimizer


def get_train_op(optimizer: Optimizer,
                 hparams: Optional[Union[HParams, Dict[str, Any]]] = None) -> \
        Callable[[torch.Tensor], None]:
    """Creates a training op..

        Args:
            loss: A scalar Tensor representing the loss to minimize.
            optimizer: A torch.optim.Optimizer instance to optimize the loss.
            hparams (dict or HParams, optional): hyperparameters. Missing
                hyperparameters are set to default values automatically. See
                :func:`~texar.core.default_optimization_hparams` for
                all hyperparameters and default values.

        Returns:
            train_op: the operator used for variables optimization.
    """
    hparams = HParams(hparams, default_optimization_hparams())

    hparams_scheduler = hparams["learning_rate_decay"]
    hparams_grad_clip = hparams["gradient_clip"]

    scheduler_type = hparams_scheduler["type"]
    if scheduler_type == "" or scheduler_type is None:
        scheduler = None
    else:
        if isinstance(scheduler_type, _LRScheduler):
            scheduler_class = scheduler_type
        else:
            scheduler_modules = ['torch.optim.lr_scheduler',
                                 'texar.custom']
            try:
                scheduler_class = utils.check_or_get_class(  # type: ignore
                    scheduler_type, scheduler_modules, _LRScheduler)
            except TypeError:
                raise ValueError(
                    "Unrecognized lr_scheduler. Must be string name of the "
                    "lr_scheduler class, or the class which is a subclass of "
                    "torch.optim._LRScheduler.")

        scheduler_kwargs = hparams_scheduler["kwargs"].todict()
        scheduler_kwargs.update({"optimizer": optimizer})
        scheduler = scheduler_class(**scheduler_kwargs)  # type: ignore

    grad_clip_type = hparams_grad_clip["type"]
    if grad_clip_type == "" or grad_clip_type is None:
        grad_clip_fn = None
    else:
        grad_clip_modules = ['torch.nn.utils',
                             'texar.custom']
        grad_clip_fn = utils.get_function(grad_clip_type, grad_clip_modules)
        grad_clip_fn_kwargs = hparams_grad_clip["kwargs"].todict()

    params_list = []

    for param_group in optimizer.param_groups:  # type: ignore
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params_list.append(params)
        else:
            params_list += params

    def _train_op(loss):
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_fn is not None:
            grad_clip_fn(parameters=params_list, **grad_clip_fn_kwargs)
        optimizer.step()
        # TODO: Ideally, scheduler should be used in the epoch level.
        if scheduler is not None:
            scheduler.step()

    return _train_op
