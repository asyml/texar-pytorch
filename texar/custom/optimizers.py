from typing import Callable, Iterable, List, Optional, Tuple, Union, Dict

import torch
from mypy_extensions import TypedDict
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer

from texar.utils.types import MaybeDict


class BertAdamParamDict(TypedDict):
    params: List[nn.Parameter]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    max_grad_norm: float


class StateDict(TypedDict):
    next_m: torch.Tensor
    next_v: torch.Tensor


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
    state: Dict[nn.Parameter, StateDict]

    def __init__(self, params: Union[MaybeDict[Iterable[nn.Parameter]]],
                 lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08, weight_decay: float = 0,
                 max_grad_norm: float = 1.0):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
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
