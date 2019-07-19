import warnings
import torch
from torch._six import inf

def l2norm(input):
    return torch.norm(input, 2)

def global_norm(t_list):
    return torch.sqrt(sum([l2norm(t)**2 for t in t_list]))

def clip_by_global_norm(parameters, clip_norm):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    use_norm = global_norm(parameters)
    return


