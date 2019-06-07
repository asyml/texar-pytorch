import math

import torch
from torch import nn


class BertGELU(nn.Module):
    """
    Bert use GELU as the activation function for the poswise network.
    """

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GPTGELU(nn.Module):
    """
    For information: OpenAI GPT's gelu is slightly different (and gives slightly
    different results).
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
