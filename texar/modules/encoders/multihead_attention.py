import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer for Sentence Blocks.
    For computational efficiency, dot-product to calculate
    query-key scores is performed in all the heads together.
    """

    def __init__(self, n_units, multi_heads=8, attention_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(n_units, n_units, bias=False)
        self.W_K = nn.Linear(n_units, n_units, bias=False)
        self.W_V = nn.Linear(n_units, n_units, bias=False)
        self.finishing_linear_layer = nn.Linear(n_units, n_units, bias=False)
        self.h = multi_heads
        self.scale_score = 1. / (n_units // multi_heads) ** 0.5
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, z=None, mask=None):
        h = self.h
        Q = self.W_Q(x)
        if z is None:
            K, V = self.W_K(x), self.W_V(x)
        else:
            K, V = self.W_K(z), self.W_V(z)

        batch, n_querys, n_units = Q.shape
        _, n_keys, _ = K.shape

        # Calculate attention scores with mask for zero-padded areas
        # Perform multi-head attention using pseudo batching all together
        # at once for efficiency
        Q = torch.cat(torch.chunk(Q, h, dim=2), dim=0)
        K = torch.cat(torch.chunk(K, h, dim=2), dim=0)
        V = torch.cat(torch.chunk(V, h, dim=2), dim=0)

        assert (Q.shape == (batch * h, n_querys, n_units // h))
        assert (K.shape == (batch * h, n_keys, n_units // h))
        assert (V.shape == (batch * h, n_keys, n_units // h))

        mask = torch.cat([mask] * h, dim=0)
        Q.mul_(self.scale_score)
        batch_A = torch.bmm(Q, K.transpose(1, 2).contiguous())

        batch_A = batch_A.masked_fill(1. - mask, -np.inf)
        batch_A = F.softmax(batch_A, dim=2)

        # Replaces 'NaN' with zeros and other values with the original ones
        batch_A = batch_A.masked_fill(batch_A != batch_A, 0.)
        assert (batch_A.shape == (batch * h, n_querys, n_keys))

        # Attention Dropout
        batch_A = self.dropout(batch_A)

        # Calculate Weighted Sum
        C = torch.bmm(batch_A, V)
        assert (C.shape == (batch * h, n_querys, n_units // h))

        # Joining the Multiple Heads
        C = torch.cat(torch.chunk(C, h, dim=0), dim=2)
        assert (C.shape == (batch, n_querys, n_units))

        # Final linear layer
        C = self.finishing_linear_layer(C)
        return C
