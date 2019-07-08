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
The XLNet model and wrappers. Adapted from
https://github.com/zihangdai/xlnet/blob/master/modeling.py
"""

import itertools
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import texar as tx
import torch
from torch import LongTensor, Tensor
from torch import nn
from torch.nn import functional as F

from xlnet.model import utils
from xlnet.model.modules import (
    PositionWiseFF, RelativeMultiheadAttention, RelativePositionalEncoding)

__all__ = [
    "XLNet",
    "XLNetClassifier",
    "XLNetRegressor",
]


class XLNet(tx.ModuleBase):
    def __init__(self, hparams=None):
        super().__init__(hparams)

        num_layers = self._hparams.num_layers
        num_heads = self._hparams.num_heads
        head_dim = self._hparams.head_dim

        self.word_embed = nn.Embedding(
            self._hparams.vocab_size, self._hparams.hidden_dim)
        self.pos_embed = RelativePositionalEncoding(
            hparams={
                "dim": self._hparams.hidden_dim,
                "max_seq_len": self._hparams.max_seq_len,
            })
        self.dropout = nn.Dropout(self._hparams.dropout)

        if self._hparams.untie_r:
            bias_shape = (num_layers, num_heads, head_dim)
        else:
            bias_shape = (num_heads, head_dim)
        self.r_w_bias = nn.Parameter(torch.Tensor(*bias_shape))
        self.r_r_bias = nn.Parameter(torch.Tensor(*bias_shape))
        self.r_s_bias = nn.Parameter(torch.Tensor(*bias_shape))

        self.attn_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        rel_attn_hparams = tx.utils.dict_fetch(
            self._hparams, RelativeMultiheadAttention.default_hparams())
        ff_hparams = tx.utils.dict_fetch(
            self._hparams, PositionWiseFF.default_hparams())
        for idx in range(num_layers):
            self.attn_layers.append(
                RelativeMultiheadAttention(hparams=rel_attn_hparams))
            self.ff_layers.append(PositionWiseFF(hparams=ff_hparams))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.r_w_bias, 0.0, 0.02)
        nn.init.normal_(self.r_r_bias, 0.0, 0.02)
        nn.init.normal_(self.r_s_bias, 0.0, 0.02)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "untie_r": True,
            "num_layers": 24,
            "mem_len": 0,
            "reuse_len": 0,
            # layer
            "num_heads": 16,
            "hidden_dim": 1024,
            "head_dim": 64,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "use_segments": True,
            # ffn
            "ffn_inner_dim": 4096,
            "activation": 'gelu',
            # embedding
            "vocab_size": 32000,
            "max_seq_len": 512,
        }

    @property
    def output_size(self):
        return self._hparams.hidden_dim

    @staticmethod
    def _cache_mem(output: Tensor, prev_mem: Optional[Tensor],
                   mem_len: int, reuse_len: int = 0) -> Tensor:
        r"""Cache hidden states into memory."""
        assert mem_len > 0

        if reuse_len is not None and reuse_len > 0:
            output = output[:reuse_len]
        if prev_mem is None:
            new_mem = output[-mem_len:]
        else:
            new_mem = torch.cat([prev_mem, output], dim=0)[-mem_len:]
        return new_mem.detach()

    def _create_mask(self, seq_len: int, mem_len: int,
                     same_length: bool = False) -> Tensor:
        r"""Create causal attention mask of shape
        `(seq_len, mem_len + seq_len)`.
        """
        device = self.r_w_bias.device
        attn_mask = torch.ones([seq_len, seq_len], device=device)
        mask_u = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros([seq_len, mem_len], device=device)
        ret = torch.cat([attn_mask_pad, mask_u], dim=1)
        if same_length:
            mask_l = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :seq_len] + mask_l, ret[:, seq_len:]], 1)
        return ret

    def __repr__(self):
        r"""Create a compressed representation by combining identical modules in
        `nn.ModuleList`s and `nn.ParameterList`s.
        """

        def _get_indent(s: str) -> int:
            return len(s) - len(s.lstrip(' '))

        class ModuleRepr(NamedTuple):
            indent: int
            repr_str: str
            names: List[str]

        def _convert_repr(module: ModuleRepr) -> List[str]:
            prefix = (f"{' ' * module.indent}(id" +
                      (f"s {module.names[0]}-{module.names[-1]}"
                       if len(module.names) > 1 else f" {module.names[0]}") +
                      "): ")
            lines = module.repr_str.split('\n')
            lines[0] = prefix + lines[0]
            return lines

        repr_str = super().__repr__().split('\n')
        # module description indexed by indent
        nested = True
        while nested:
            nested = False
            output_str = []
            prev_module: Optional[ModuleRepr] = None
            for idx, line in enumerate(repr_str):
                line = repr_str[idx]
                indent = _get_indent(line)
                if prev_module is not None and prev_module.indent > indent:
                    output_str.extend(_convert_repr(prev_module))
                    prev_module = None
                name = line[(indent + 1):line.find(')')]
                if line[indent] != '(' or not name.isnumeric():
                    if prev_module is None:
                        output_str.append(line)
                    continue

                end_idx = next(
                    end_idx for end_idx in range(idx + 1, len(repr_str))
                    if _get_indent(repr_str[end_idx]) <= indent)
                end_indent = _get_indent(repr_str[end_idx])
                if end_indent < indent or repr_str[end_idx][end_indent] != ')':
                    # not a module; a parameter in ParameterList
                    end_idx -= 1
                    indent -= 2  # parameters are somehow indented further
                module_repr = '\n'.join(
                    [line[(indent + len(name) + 4):]] +  # "(): "
                    repr_str[(idx + 1):(end_idx + 1)])
                if prev_module is None:
                    prev_module = ModuleRepr(indent, module_repr, [name])
                elif prev_module.indent < indent:
                    nested = True
                elif prev_module.repr_str == module_repr:
                    prev_module.names.append(name)
                else:
                    output_str.extend(_convert_repr(prev_module))
                    prev_module = ModuleRepr(indent, module_repr, [name])
            repr_str = output_str
        return '\n'.join(repr_str)

    def forward(self, token_ids: LongTensor, segment_ids: Optional[LongTensor],
                input_mask: Optional[Tensor] = None,
                memory: Optional[List[Tensor]] = None,
                permute_mask: Optional[Tensor] = None,
                bi_data: bool = False, clamp_len: Optional[int] = None,
                same_length: bool = False, attn_type: str = 'bi') \
            -> Tuple[Tensor, Optional[List[Tensor]]]:
        r"""

        Args:
            token_ids: Shape `(seq_len, batch_size)`.
            segment_ids: Shape `(seq_len, batch_size)`.
            input_mask: Float tensor of shape `(seq_len, batch_size)`. Note that
                positions with value 1 are masked out.
            memory: Memory from previous batches. A list of length `num_layers`,
                each a tensor of shape `(mem_len, batch_size, hidden_dim)`.
            permute_mask: The permutation mask. Float tensor of shape
                `(seq_len, seq_len, batch_size)`.
                A value of 0 for ``permute_mask[i, j, k]`` indicates that
                position `i` attends to position `j` in batch `k`.
            bi_data (bool): Whether to use bidirectional data input pipeline.
            clamp_len (int): Clamp all relative distances larger than
                :attr:`clamp_len`. A value of -1 means no clamping.
            same_length (bool): Whether to use the same attention length for
                each token.
            attn_type (str): Attention type. Supported values are `"uni"`
                and `"bi"`.

        :returns: A tuple of `(output, new_memory)`:

            - **`output`**: The final layer output representations. Shape
              `(seq_len, batch_size, hidden_dim)`.
            - **`new_memory`**: The memory of the current batch.
              If `mem_len` is 0, then `new_memory` is `None`. Otherwise, it is a
              list of length `num_layers`, each a tensor of shape
              `(mem_len, batch_size, hidden_dim)`.
              This can be used as the :attr:`memory` argument in the next batch.
        """
        seq_len, batch_size = token_ids.size()
        mem_len = memory[0].size(0) if memory is not None else 0
        tot_len = seq_len + mem_len
        reuse_len = self._hparams.reuse_len

        # Construct masks.
        masks: List[Tensor] = []

        # Causal attention mask.
        if attn_type == 'uni':
            attn_mask = self._create_mask(seq_len, mem_len, same_length)
            # attn_mask: (seq_len, tot_len, 1, 1)
            attn_mask = attn_mask.unsqueeze(2).unsqueeze(3)
            masks.append(attn_mask)
        elif attn_type == 'bi':
            pass
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        # Data mask: input mask & permutation mask.
        data_mask = utils.sum_tensors([input_mask, permute_mask])
        if data_mask is not None:
            # All positions in memory can be attended to.
            if data_mask.dim() == 2:
                data_mask = data_mask.unsqueeze(0)
            memory_mask = data_mask.new_zeros(
                data_mask.size(0), mem_len, batch_size)
            # data_mask: (seq_len, tot_len, batch_size, 1)
            data_mask = torch.cat([memory_mask, data_mask], dim=1).unsqueeze(3)
            masks.append(data_mask)

        # Exclude the main diagonal (target tokens) from the mask.
        attn_mask = utils.sum_tensors(masks)
        if attn_mask is not None:
            attn_mask = (attn_mask > 0).float()
            final_mask = -torch.eye(seq_len, device=attn_mask.device)
            final_mask = torch.cat([
                final_mask.new_zeros(seq_len, mem_len), final_mask], dim=-1)
            final_mask = final_mask.unsqueeze(2).unsqueeze(3)
            # final_mask: (seq_len, tot_len, batch_size, 1)
            final_mask = ((attn_mask + final_mask) > 0)
        else:
            final_mask = None

        # Construct segment embedding.
        if segment_ids is not None:
            concat_segment_ids = torch.cat([
                segment_ids.new_zeros(mem_len, batch_size), segment_ids])
            segment_matrix = (segment_ids.unsqueeze(1) !=
                              concat_segment_ids.unsqueeze(0)).long()
            segment_matrix = F.one_hot(segment_matrix, num_classes=2).float()
        else:
            segment_matrix = None

        word_embed = self.word_embed(token_ids)

        pos_embed = self.pos_embed(
            batch_size, seq_len, tot_len, clamp_len, attn_type, bi_data)
        pos_embed = self.dropout(pos_embed)

        states_h = self.dropout(word_embed)
        if memory is None:
            memory = [None] * self._hparams.num_layers
        new_memory = []

        untie_r = self._hparams.untie_r
        for idx in range(self._hparams.num_layers):
            cur_memory = memory[idx] if memory is not None else None
            if mem_len > 0:
                new_memory.append(self._cache_mem(
                    states_h, cur_memory, mem_len, reuse_len))
            r_w_bias = self.r_w_bias if not untie_r else self.r_w_bias[idx]
            r_r_bias = self.r_r_bias if not untie_r else self.r_r_bias[idx]
            if segment_ids is None:
                r_s_bias = None
            else:
                r_s_bias = self.r_s_bias if not untie_r else self.r_s_bias[idx]
            states_h = self.attn_layers[idx](
                states_h, pos_embed, segment_matrix, r_w_bias, r_r_bias,
                r_s_bias, final_mask, cur_memory)
            states_h = self.ff_layers[idx](states_h)

        output = self.dropout(states_h)
        if mem_len == 0:
            return output, None
        return output, new_memory


class XLNetSummary(tx.ModuleBase):
    hidden_to_logits: nn.Linear

    def __init__(self, hparams=None):
        super().__init__(hparams)

        xlnet = XLNet(self._hparams.xlnet)
        self.xlnet = xlnet
        if self._hparams.use_projection:
            self.projection = nn.Linear(xlnet.output_size, xlnet.output_size)
        self.dropout = nn.Dropout(self._hparams.xlnet.dropout)

        if self._hparams.summary_type == 'last':
            self.summary_op = lambda output: output[-1]
        elif self._hparams.summary_type == 'first':
            self.summary_op = lambda output: output[0]
        elif self._hparams.summary_type == 'mean':
            self.summary_op = lambda output: torch.mean(output, dim=0)
        elif self._hparams.summary_type == 'attn':
            raise NotImplementedError
        else:
            raise ValueError(
                f"Unsupported summary type {self._hparams.summary_type}")

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "xlnet": XLNet.default_hparams(),
            "summary_type": "last",  # "first", "mean", "attn"
            "use_projection": True,
        }

    def param_groups(self, lr: Optional[float] = None,
                     lr_layer_scale: float = 1.0,
                     decay_base_params: bool = False):
        r"""Create parameter groups for optimizers. When
        :attr:`lr_layer_decay_rate` is not 1.0, parameters from each layer form
        separate groups with different base learning rates.

        Args:
            lr (float): The learning rate. Can be omitted if
                :attr:`lr_layer_decay_rate` is 1.0.
            lr_layer_scale (float): Per-layer LR scaling rate. The `i`-th layer
                will be scaled by `lr_layer_scale ^ (num_layers - i - 1)`.
            decay_base_params (bool): If `True`, treat non-layer parameters
                (e.g. embeddings) as if they're in layer 0. If `False`, these
                parameters are not scaled.

        Returns:
            The parameter groups, used as the first argument for optimizers.
        """

        if lr_layer_scale != 1.0:
            if lr is None:
                raise ValueError(
                    "lr must be specified when lr_layer_decay_rate is not 1.0")

            def params_except_in(module: nn.Module,
                                 except_names: List[str]) \
                    -> Iterable[nn.Parameter]:
                return itertools.chain.from_iterable(
                    child.parameters() for name, child in
                    module.named_children()
                    if name not in except_names)

            num_layers = self._hparams.xlnet.num_layers
            base_group = {
                "params": params_except_in(
                    self.xlnet, ['attn_layers', 'ff_layers']),
                "lr": lr * (lr_layer_scale ** num_layers
                            if decay_base_params else 1.0)
            }
            fine_tune_group = {
                "params": params_except_in(self, ["xlnet"]),
                "lr": lr
            }
            param_groups = [base_group, fine_tune_group]
            for idx in range(num_layers):
                decay_rate = lr_layer_scale ** (num_layers - idx - 1)
                param_group = {
                    "params": [*self.xlnet.attn_layers[idx].parameters(),
                               *self.xlnet.ff_layers[idx].parameters()],
                    "lr": lr * decay_rate,
                }
                param_groups.append(param_group)
        else:
            param_groups = self.parameters()
        return param_groups

    def forward(self, token_ids: LongTensor, segment_ids: Optional[LongTensor],
                input_mask: Optional[Tensor] = None) -> Tensor:
        # output: (seq_len, batch_size, hidden_dim)
        output, _ = self.xlnet(token_ids, segment_ids, input_mask=input_mask)
        summary = self.summary_op(output)
        if self._hparams.use_projection:
            summary = torch.tanh(self.projection(summary))
        # summary: (batch_size, hidden_dim)
        summary = self.dropout(summary)
        return summary


class XLNetClassifier(XLNetSummary, tx.modules.ClassifierBase):
    r"""An encapsulated XLNet model used for classification tasks."""

    def __init__(self, hparams=None):
        super().__init__(hparams)

        self.hidden_to_logits = nn.Linear(
            self.xlnet.output_size, self._hparams.num_classes)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            **XLNetSummary.default_hparams(),
            "num_classes": 2,
        }

    def forward(self, token_ids: LongTensor, segment_ids: Optional[LongTensor],
                labels: LongTensor, input_mask: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:
        summary = super().forward(token_ids, segment_ids, input_mask)
        # logits: (batch_size, num_classes)
        logits = self.hidden_to_logits(summary)
        # loss: (batch_size)
        loss = F.cross_entropy(logits, labels.view(-1), reduction='none')
        return loss, logits


class XLNetRegressor(XLNetSummary, tx.ModuleBase):
    r"""An encapsulated XLNet model used for regression tasks."""

    def __init__(self, hparams=None):
        super().__init__(hparams)

        self.hidden_to_logits = nn.Linear(self.xlnet.output_size, 1)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return XLNetSummary.default_hparams()

    def forward(self, token_ids: LongTensor, segment_ids: Optional[LongTensor],
                labels: Tensor, input_mask: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:
        summary = super().forward(token_ids, segment_ids, input_mask)
        # logits: (batch_size)
        logits = self.hidden_to_logits(summary).squeeze(-1)
        # loss: (batch_size)
        loss = (logits - labels.view(-1)) ** 2
        return loss, logits
