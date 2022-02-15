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

import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from mypy_extensions import TypedDict

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from texar.torch.hyperparams import HParams
from texar.torch.utils import utils
from texar.torch.utils.types import MaybeList

__all__ = [
    "default_optimization_hparams",
    "get_optimizer",
    "get_scheduler",
    "get_grad_clip_fn",
    "get_train_op",
    "BertAdam"
]


def default_optimization_hparams() -> Dict[str, Any]:
    r"""Returns a `dict` of default hyperparameters of training op
    and their default values

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

    `"optimizer"`: dict
        Hyperparameters of a
        :torch_docs:`torch.optim.Optimizer <optim.html#torch.optim.Optimizer>`.

        - `"type"` specifies the optimizer class. This can be

          - The string name or full module path of an optimizer class.
            If the class name is provided, the class must be in module
            :torch_docs:`torch.optim <optim.html>` or :mod:`texar.torch.custom`,
            :mod:`texar.torch.core.optimization`
          - An optimizer class.
          - An instance of an optimizer class.

          For example

          .. code-block:: python

              "type": "Adam"                    # class name
              "type": "my_module.MyOptimizer"   # module path
              "type": texar.torch.custom.BertAdam     # class
              "type": my_module.MyOptimizer     # class

        - `"kwargs"` is a `dict` specifying keyword arguments for creating
          the optimizer class instance, with :python:`opt_class(**kwargs)`.
          Ignored if `"type"` is a class instance.

    `"learning_rate_decay"`: dict
        Hyperparameters of learning rate decay function. The learning rate
        starts decay from :attr:`"start_decay_step"` and keeps unchanged after
        :attr:`"end_decay_step"` or reaching :attr:`"min_learning_rate"`.

        The decay function is specified in `"type"` and `"kwargs"`.

        - `"type"` can be a decay function or its name or module path. If
          function name is provided, it must be from module
          :torch_docs:`torch.optim <optim.html>` or :mod:`texar.torch.custom`,
          :mod:`texar.torch.core.optimization`.

        - `"kwargs"` is a `dict` of keyword arguments for the function
          excluding arguments named `"global_step"` and `"learning_rate"`.

        The function is called with
        :python:`lr = decay_fn(learning_rate=lr, global_step=offset_step,
        **kwargs)`, where `offset_step` is the global step offset as above.

    `"gradient_clip"`: dict
        Hyperparameters of gradient clipping. The gradient clipping function
        takes a list of `(gradients, variables)` tuples and returns a list
        of `(clipped_gradients, variables)` tuples. Typical examples include
        :torch_nn:`utils.clip_grad_norm_` and
        :torch_nn:`utils.clip_grad_value_`.

        "type" specifies the gradient clip function, and can be a function,
        or its name or module path. If function name is provided, the
        function must be from module :mod:`torch.nn.utils`,
        :mod:`texar.torch.custom`, or :mod:`texar.torch.core.optimization`.

        `"kwargs"` specifies keyword arguments to the function, except arguments
        named `"parameters"`.

    `"gradient_noise_scale"`: float, optional
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
        params: Iterable[Union[torch.Tensor, Dict[str, Any]]],
        hparams: Optional[Union[HParams, Dict[str, Any]]] = None) -> \
        Optimizer:
    r"""Creates a optimizer instance.

    Args:
        params: an iterable of :class:`torch.Tensor` or
            :class:`dict`. Specifies what Tensors should be optimized.
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically. See
            :func:`~texar.torch.core.default_optimization_hparams` for
            all hyperparameters and default values.

    :return:
        The :torch_docs:`torch.optim.Optimizer
        <optim.html#torch.optim.Optimizer>` instance specified in
        :attr:`hparams`.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_optimization_hparams())

    hparams_opt = hparams["optimizer"]

    optimizer_type = hparams_opt["type"]
    if isinstance(optimizer_type, Optimizer):
        optimizer_class = optimizer_type
    else:
        optimizer_modules = ['torch.optim',
                             'texar.torch.custom']
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


def get_scheduler(optimizer: Optimizer,
                  hparams: Optional[Union[HParams, Dict[str, Any]]] = None) -> \
        Optional[_LRScheduler]:
    r"""Creates a scheduler instance.

    Args:
        optimizer: A :torch_docs:`torch.optim.Optimizer
            <optim.html#torch.optim.Optimizer>` instance.
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically. See
            :func:`~texar.torch.core.default_optimization_hparams` for
            all hyperparameters and default values.

    :return:
        A :torch_docs:`torch.optim.lr_scheduler._LRScheduler
        <optim.html#how-to-adjust-learning-rate>` instance.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_optimization_hparams())

    hparams_scheduler = hparams["learning_rate_decay"]

    scheduler_type = hparams_scheduler["type"]
    if scheduler_type == "" or scheduler_type is None:
        scheduler = None
    else:
        if isinstance(scheduler_type, _LRScheduler):
            scheduler_class = scheduler_type
        else:
            scheduler_modules = ['torch.optim.lr_scheduler',
                                 'texar.torch.custom']
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

    return scheduler


def get_grad_clip_fn(hparams: Optional[Union[HParams,
                                             Dict[str, Any]]] = None) -> \
        Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]]:
    r"""Create a gradient clipping function.

    Args:
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically. See
            :func:`~texar.torch.core.default_optimization_hparams` for
            all hyperparameters and default values.

    Returns:
        A gradient clipping function.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_optimization_hparams())

    hparams_grad_clip = hparams["gradient_clip"]

    grad_clip_type = hparams_grad_clip["type"]
    if grad_clip_type == "" or grad_clip_type is None:
        grad_clip_fn = None
    else:
        grad_clip_modules = ['torch.nn.utils',
                             'texar.torch.custom']
        grad_clip_fn = utils.get_function(grad_clip_type, grad_clip_modules)
        grad_clip_fn_kwargs = hparams_grad_clip["kwargs"].todict()
        grad_clip_fn = functools.partial(grad_clip_fn, **grad_clip_fn_kwargs)

    return grad_clip_fn


def get_train_op(params: Optional[Iterable[Union[torch.Tensor,
                                                 Dict[str, Any]]]] = None,
                 optimizer: Optional[Optimizer] = None,
                 scheduler: Optional[_LRScheduler] = None,
                 hparams: Optional[Union[HParams, Dict[str, Any]]] = None) -> \
        Callable[[], None]:
    r"""Creates a training op.

    Args:
        params: an iterable of :class:`torch.Tensor` or
            :class:`dict`. Specifies what Tensors should be optimized.
        optimizer: A :torch_docs:`torch.optim.Optimizer
            <optim.html#torch.optim.Optimizer>` instance.
        scheduler: A :torch_docs:`torch.optim.lr_scheduler._LRScheduler
            <optim.html#how-to-adjust-learning-rate>` instance.
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically. See
            :func:`~texar.torch.core.default_optimization_hparams` for
            all hyperparameters and default values.

    Returns:
        The callable used for variable optimization.
    """
    hparams = HParams(hparams, default_optimization_hparams())

    if params is None and optimizer is None and scheduler is None:
        raise ValueError("'params', 'optimizer' and 'scheduler' must not be "
                         "None simultaneously.")

    if scheduler is None:
        if optimizer is None and params is not None:
            optimizer = get_optimizer(params, hparams)
        if optimizer is not None:
            scheduler = get_scheduler(optimizer, hparams)
    else:
        optimizer = scheduler.optimizer  # type: ignore

    grad_clip_fn = get_grad_clip_fn(hparams)

    # TODO: Support per-parameter options in the future.
    params_list: List[nn.Parameter] = []
    for param_group in optimizer.param_groups:  # type: ignore
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params_list.append(params)
        elif isinstance(params, list):
            params_list += params

    def _train_op():
        if grad_clip_fn is not None:
            grad_clip_fn(parameters=params_list)
        optimizer.step()
        # TODO: Ideally, scheduler should be used in the epoch level.
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return _train_op


class BertAdamParamDict(TypedDict):
    r"""The :attr:`param_groups` dictionary used in PyTorch optimizers."""
    params: List[nn.Parameter]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    max_grad_norm: float


class BertAdamStateDict(TypedDict):
    r"""The :attr:`state` dictionary used in :class:`BertAdam` optimizer."""
    next_m: torch.Tensor
    next_v: torch.Tensor


OptimParamType = Union[
    MaybeList[Iterable[nn.Parameter]],  # model.parameters()
    MaybeList[Dict[str, Any]],  # {"params": ..., "other_kwargs": ...}
]


class BertAdam(Optimizer):
    r"""Implements BERT version of Adam algorithm with weight decay fix.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping).
            Default: 1.0
    """

    param_groups: List[BertAdamParamDict]
    state: Dict[nn.Parameter, BertAdamStateDict]

    def __init__(self, params: OptimParamType,
                 lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08, weight_decay: float = 0,
                 max_grad_norm: float = 1.0):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)  # type: ignore

    def step(self, closure: Optional[Callable[[], float]] = None):
        r"""Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please "
                        "consider SparseAdam instead")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['betas']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['eps'])

                # Just adding the square of the weights to the loss function is
                # *not* # the correct way of using L2 regularization or weight
                # decay with Adam, since that will interact with the m and v
                # parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't
                # interact with the m/v parameters. This is equivalent to adding
                # the square of the weights to the loss with plain
                # (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                lr = group['lr']
                update_with_lr = lr * update
                p.data.add_(-update_with_lr)

                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss
