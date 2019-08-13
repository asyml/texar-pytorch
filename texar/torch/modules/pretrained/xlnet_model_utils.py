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
Model Utils of XLNet Modules.

Adapted from
https://github.com/zihangdai/xlnet/blob/master/modeling.py
"""
import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from texar.torch.core import get_layer
from texar.torch.module_base import ModuleBase

__all__ = [
    "PositionWiseFF",
    "RelativeMultiheadAttention",
    "RelativePositionalEncoding",
    "params_except_in",
    "init_weights",
]


class PositionWiseFF(ModuleBase):

    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        hidden_dim = self._hparams.hidden_dim
        ffn_inner_dim = self._hparams.ffn_inner_dim
        dropout = self._hparams.dropout
        activation = self._hparams.activation.capitalize()
        if activation == 'Relu':
            activation = 'ReLU'
        elif activation == 'Gelu':
            activation = 'GPTGELU'

        self.linear1 = nn.Linear(hidden_dim, ffn_inner_dim)
        self.activation_fn = get_layer({"type": activation})
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(ffn_inner_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "hidden_dim": 768,
            "ffn_inner_dim": 3072,
            "dropout": 0.1,
            "activation": 'relu',
        }

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output.
        """
        return self._hparams.hidden_dim

    def forward(self,  # type: ignore
                input: torch.Tensor) -> torch.Tensor:
        # position-wise feed-forward
        output = self.linear1(input)
        output = self.activation_fn(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)
        # residual + layer norm
        output = self.layer_norm(input + output)
        return output


class PositionalEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, embed_dim: int):
        super().__init__()

        freq_seq = torch.arange(0.0, embed_dim, 2.0)
        inv_freq = 1 / (10000 ** (freq_seq / embed_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self,  # type: ignore
                pos_seq: torch.Tensor) -> torch.Tensor:
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        pos_embed = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return pos_embed


class RelativePositionalEncoding(ModuleBase):

    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)
        # self.sinusoid_embed = tx.modules.SinusoidsPositionEmbedder(
        #     None, hparams={
        #         "dim": self._hparams.dim,
        #         "cache_embeddings": False,
        #     })
        self.sinusoid_embed = PositionalEmbedding(self._hparams.dim)

    @staticmethod
    def default_hparams():
        return {
            "dim": 768,
            "max_seq_len": 512,
        }

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output.
        """
        return self._hparams.dim

    def _create_positional_embedding(self,
                                     start: int,
                                     end: int,
                                     step: int,
                                     batch_size: int,
                                     clamp_len: Optional[int] = None) \
            -> torch.Tensor:
        embed_buffer = next(self.sinusoid_embed.buffers())
        pos_seq = torch.arange(start, end, step, device=embed_buffer.device,
                               dtype=embed_buffer.dtype)

        if clamp_len is not None:
            pos_seq = torch.clamp(pos_seq, -clamp_len, clamp_len)

        pos_embed = self.sinusoid_embed(pos_seq)
        pos_embed = pos_embed.unsqueeze(1).expand(-1, batch_size, -1)
        return pos_embed

    def forward(self,  # type: ignore
                batch_size: int,
                seq_len: int,
                total_len: int,
                clamp_len: Optional[int] = None,
                attn_type: str = 'bi',
                bi_data: bool = True) -> torch.Tensor:
        if attn_type == 'bi':
            start, end = total_len, -seq_len
        elif attn_type == 'uni':
            start, end = total_len, -1
        else:
            raise ValueError(f"Unknown `attn_type` {attn_type}")

        if bi_data:
            if batch_size % 2 != 0:
                raise ValueError("`batch_size` must be an even number")
            fwd_pos_embed = self._create_positional_embedding(
                start, end, -1, batch_size // 2, clamp_len)
            bwd_pos_embed = self._create_positional_embedding(
                -start, -end, 1, batch_size // 2, clamp_len)
            pos_embed = torch.cat([fwd_pos_embed, bwd_pos_embed], dim=1)
        else:
            pos_embed = self._create_positional_embedding(
                start, end, -1, batch_size, clamp_len)
        return pos_embed


class RelativeMultiheadAttention(ModuleBase):
    def __init__(self,
                 r_r_bias: Optional[nn.Parameter] = None,
                 r_w_bias: Optional[nn.Parameter] = None,
                 r_s_bias: Optional[nn.Parameter] = None,
                 hparams=None):
        super().__init__(hparams=hparams)

        self.num_heads = self._hparams.num_heads
        self.head_dim = self._hparams.head_dim
        hidden_dim = self._hparams.hidden_dim

        self.head_projection = nn.Linear(
            hidden_dim, 3 * self.num_heads * self.head_dim, bias=False)
        self.pos_projection = nn.Linear(
            hidden_dim, self.num_heads * self.head_dim, bias=False)

        self.dropout = nn.Dropout(self._hparams.dropout)
        self.dropout_attn = nn.Dropout(self._hparams.attention_dropout)
        self.output_projection = nn.Linear(
            self.num_heads * self.head_dim, hidden_dim, bias=False)

        bias_shape = (self.num_heads, self.head_dim)
        self.untie_r = r_r_bias is None
        self.r_r_bias = (r_r_bias if r_r_bias is not None
                         else nn.Parameter(torch.Tensor(*bias_shape)))
        self.r_w_bias = (r_w_bias if r_w_bias is not None
                         else nn.Parameter(torch.Tensor(*bias_shape)))

        if self._hparams.use_segments:
            self.segment_embed = nn.Parameter(torch.Tensor(
                2, self.num_heads, self.head_dim))
            self.r_s_bias = (r_s_bias if r_s_bias is not None
                             else nn.Parameter(torch.Tensor(*bias_shape)))

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)

        self.scale = 1 / (self.head_dim ** 0.5)
        self.reset_parameters()

    def reset_parameters(self):
        if self.untie_r:
            nn.init.normal_(self.r_w_bias, 0.0, 0.02)
            nn.init.normal_(self.r_r_bias, 0.0, 0.02)
        if self._hparams.use_segments:
            nn.init.normal_(self.segment_embed, 0.0, 0.02)
            if self.untie_r:
                nn.init.normal_(self.r_s_bias, 0.0, 0.02)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "num_heads": 12,
            "hidden_dim": 768,
            "head_dim": 64,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "use_segments": True,
        }

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output
        :attr:`output_h`.
        """
        return self._hparams.hidden_dim

    @staticmethod
    def _rel_shift(x: torch.Tensor, klen: int) -> torch.Tensor:
        shape = x.size()
        x = x.view(shape[1], shape[0], *shape[2:])[1:]
        x = x.view(shape[0], shape[1] - 1, *shape[2:])[:, :klen]
        return x

    def _compute_attention_score(self,
                                 q_head: torch.Tensor,
                                 k_head_h: torch.Tensor,
                                 v_head_h: torch.Tensor,
                                 k_head_r: torch.Tensor,
                                 segment_mat: Optional[torch.Tensor] = None,
                                 attn_mask: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        # Content based attention score.
        q_head_rw = q_head + self.r_w_bias
        # attn_ac: (seq_len, tot_len, batch_size, n_head)
        attn_ac = torch.einsum('ibnd,jbnd->ijbn', [q_head_rw, k_head_h])

        # Position based attention score.
        q_head_rr = q_head + self.r_r_bias
        # attn_bd: (seq_len, tot_len, batch_size, n_head)
        attn_bd = torch.einsum('ibnd,jbnd->ijbn', [q_head_rr, k_head_r])
        attn_bd = self._rel_shift(attn_bd, klen=attn_ac.size(1))

        # Segment based attention score.
        if segment_mat is None:
            attn_ef = 0
        else:
            q_head_rs = q_head + self.r_s_bias
            attn_ef = torch.einsum(
                'ibnd,snd->ibns', [q_head_rs, self.segment_embed])
            attn_ef = torch.einsum('ijbs,ibns->ijbn', [segment_mat, attn_ef])

        # Merge attention scores and perform masking.
        # attn_score: (seq_len, tot_len, batch_size, n_head)
        attn_score = attn_ac + attn_bd + attn_ef
        attn_score.mul_(self.scale)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, :, :, None]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, :, :, None]
            attn_score = attn_score.float().masked_fill(
                attn_mask, -1e30).type_as(attn_score)

        # Compute attention probability.
        # attn_prob: (seq_len, tot_len, batch_size, n_head)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout_attn(attn_prob)

        # Compute attention vector.
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', [attn_prob, v_head_h])
        return attn_vec.contiguous()

    def _post_attention(self, attn_vec: torch.Tensor) -> torch.Tensor:
        attn_vec = attn_vec.view(*attn_vec.size()[:2], -1)
        attn_out = self.output_projection(attn_vec)
        attn_out = self.dropout(attn_out)
        return attn_out

    def forward(self,  # type: ignore
                states_h: torch.Tensor,
                pos_embed: torch.Tensor,
                states_g: Optional[torch.Tensor] = None,
                segment_mat: Optional[torch.Tensor] = None,
                attn_mask_h: Optional[torch.Tensor] = None,
                attn_mask_g: Optional[torch.Tensor] = None,
                target_mapping: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_len, batch_size = states_h.size()[:2]
        pos_len = pos_embed.size(0)

        if memory is not None and memory.dim() > 1:
            concat_input = torch.cat([memory, states_h], dim=0)
        else:
            concat_input = states_h

        # Content heads.
        heads = self.head_projection(concat_input)
        q_head_h, k_head_h, v_head_h = torch.chunk(heads, 3, dim=-1)
        q_head_h = q_head_h[-seq_len:]
        tot_len = k_head_h.size(0)

        q_head_h = q_head_h.view(
            seq_len, batch_size, self.num_heads, self.head_dim)
        k_head_h = k_head_h.view(
            tot_len, batch_size, self.num_heads, self.head_dim)
        v_head_h = v_head_h.view(
            tot_len, batch_size, self.num_heads, self.head_dim)

        # Positional heads.
        k_head_r = self.pos_projection(pos_embed)
        k_head_r = k_head_r.view(
            pos_len, batch_size, self.num_heads, self.head_dim)

        # Core attention ops.
        attn_vec_h = self._compute_attention_score(
            q_head_h, k_head_h, v_head_h, k_head_r,
            segment_mat, attn_mask_h)

        # Post attention processing.
        attn_out_h = self._post_attention(attn_vec_h)
        # residual + layer norm
        output_h = self.layer_norm(states_h + attn_out_h)

        output_g = None
        if states_g is not None:
            proj_dim = self.num_heads * self.head_dim
            proj_weight = self.head_projection.weight[:proj_dim]
            q_head_g = F.linear(states_g, proj_weight)
            q_head_g = q_head_g.view(
                q_head_g.size(0), batch_size, self.num_heads, self.head_dim)
            if target_mapping is not None:
                q_head_g = torch.einsum(
                    'mbnd,mlb->lbnd', [q_head_g, target_mapping])
            attn_vec_g = self._compute_attention_score(
                q_head_g, k_head_h, v_head_h, k_head_r,
                segment_mat, attn_mask_g)
            if target_mapping is not None:
                attn_vec_g = torch.einsum(
                    'lbnd,mlb->mbnd', [attn_vec_g, target_mapping])
            attn_out_g = self._post_attention(attn_vec_g)
            output_g = self.layer_norm(states_g + attn_out_g)

        return output_h, output_g


def params_except_in(module: nn.Module,
                     except_names: List[str]) \
        -> Iterable[nn.Parameter]:
    return itertools.chain.from_iterable(
        child.parameters() for name, child in
        module.named_children()
        if name not in except_names)


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, 0.02)
