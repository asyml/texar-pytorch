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
XLNet encoders.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from texar.core import layers
from texar.hyperparams import HParams
from texar.modules.pretrained import (xlnet_utils, XLNetBase, PositionWiseFF,
                                      RelativePositionalEncoding,
                                      RelativeMultiheadAttention)
from texar.utils import dict_fetch


__all__ = [
    "XLNetEncoder",
]


class XLNetEncoder(XLNetBase):
    r"""Raw XLNet module for encoding sequences.

    This module supports the architecture first proposed
    in `(Yang et al.)` XLNet.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model to load selected in the list of:
            `xlnet-base-cased`, `xlnet-large-cased`.
            If `None`, will use the model name in :attr:`hparams`.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
        init (optional): whether to initialize `XLNetEncoder`.
    """

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None,
                 init=True):

        super().__init__(pretrained_model_name=pretrained_model_name,
                         cache_dir=cache_dir,
                         hparams=hparams)

        if self.pretrained_model_dir:
            self._hparams = HParams(self.pretrained_model_hparams,
                                    self._hparams.todict())

        num_layers = self._hparams.num_layers
        num_heads = self._hparams.num_heads
        head_dim = self._hparams.head_dim

        self.word_embed = nn.Embedding(self._hparams.vocab_size,
                                       self._hparams.hidden_dim)
        self.pos_embed = RelativePositionalEncoding(
            hparams={
                "dim": self._hparams.hidden_dim,
                "max_seq_len": self._hparams.max_seq_len,
            })
        self.dropout = nn.Dropout(self._hparams.dropout)

        self.r_r_bias: Optional[nn.Parameter]
        self.r_w_bias: Optional[nn.Parameter]
        self.r_s_bias: Optional[nn.Parameter]

        if not self._hparams.untie_r:
            self.r_r_bias = nn.Parameter(torch.Tensor(num_heads, head_dim))
            self.r_w_bias = nn.Parameter(torch.Tensor(num_heads, head_dim))
            self.r_s_bias = (nn.Parameter(torch.Tensor(num_heads, head_dim))
                             if self._hparams.use_segments else None)
        else:
            self.r_r_bias = None
            self.r_w_bias = None
            self.r_s_bias = None

        self.attn_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        rel_attn_hparams = dict_fetch(
            self._hparams, RelativeMultiheadAttention.default_hparams())
        ff_hparams = dict_fetch(
            self._hparams, PositionWiseFF.default_hparams())
        for _ in range(num_layers):
            self.attn_layers.append(RelativeMultiheadAttention(
                self.r_r_bias, self.r_w_bias, self.r_s_bias,
                hparams=rel_attn_hparams))
            self.ff_layers.append(PositionWiseFF(hparams=ff_hparams))

        self.mask_emb = nn.Parameter(
            torch.Tensor(1, 1, self._hparams.hidden_dim))

        if init:
            if self.pretrained_model_dir:
                xlnet_utils.init_xlnet_checkpoint(self,
                                                  self.pretrained_model_dir)
            elif self._hparams.initializer:
                initialize = layers.get_initializer(self._hparams.initializer)
                assert initialize is not None
                # Do not re-initialize LayerNorm modules.
                for name, param in self.named_parameters():
                    if name.split('.')[-1] == 'weight' \
                            and 'layer_norm' not in name:
                        initialize(param)
            else:
                self.reset_parameters()

    def reset_parameters(self):
        if not self._hparams.untie_r:
            nn.init.normal_(self.r_w_bias, 0.0, 0.02)
            nn.init.normal_(self.r_r_bias, 0.0, 0.02)
            if self._hparams.use_segments:
                nn.init.normal_(self.r_s_bias, 0.0, 0.02)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        * The encoder arch is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the encoder arch is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the encoder arch is defined by the
          configurations in `hparams` and weights are randomly initialized.

        .. code-block:: python

            {
                "pretrained_model_name": "xlnet-base-cased",
                "untie_r": True,
                "num_layers": 12,
                "mem_len": 0,
                "reuse_len": 0,
                "num_heads": 12,
                "hidden_dim": 768,
                "head_dim": 64,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "use_segments": True,
                "ffn_inner_dim": 3072,
                "activation": 'gelu',
                "vocab_size": 32000,
                "max_seq_len": 512,
                "initializer": None,
                "name": "xlnet_encoder",
            }

        Here:

        The default parameters are values for cased XLNet-Base model.

        `pretrained_model_name`: str or None
            The name of the pre-trained XLNet model. If None, the model
            will be randomly initialized.

        `untie_r`: bool
            Whether to untie the biases in attention.

        `num_layers`: int
            The number of stacked layers.

        `mem_len`: int
            The number of tokens to cache.

        `reuse_len`: int
            The number of tokens in the current batch to be cached and reused
            in the future.

        `num_heads`: int
            The number of attention heads.

        `hidden_dim`: int
            The hidden size.

        `head_dim`: int
            The dimension size of each attention head.

        `dropout`: float
            Dropout rate.

        `attention_dropout`: float
            Dropout rate on attention probabilities.

        `use_segments`: bool
            Whether to use segment embedding.

        `ffn_inner_dim`: int
            The hidden size in feed-forward layers.

        `activation`: str
            `relu` or `gelu`.

        `vocab_size`: int
            The vocabulary size.

        `max_seq_len`: int
            The maximum sequence length for `RelativePositionalEncoding`.

        `initializer`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        `name`: str
            Name of the module.
        """

        return {
            'pretrained_model_name': 'xlnet-base-cased',
            'untie_r': True,
            'num_layers': 12,
            'mem_len': 0,
            'reuse_len': 0,
            # layer
            'num_heads': 12,
            'hidden_dim': 768,
            'head_dim': 64,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'use_segments': True,
            # ffn
            'ffn_inner_dim': 3072,
            'activation': 'gelu',
            # embedding
            'vocab_size': 32000,
            'max_seq_len': 512,
            'initializer': None,
            'name': "xlnet_encoder",
            '@no_typecheck': ['pretrained_model_name'],
        }

    @property
    def output_size(self):
        return self._hparams.hidden_dim

    @staticmethod
    def _cache_mem(output: torch.Tensor,
                   prev_mem: Optional[torch.Tensor],
                   mem_len: int,
                   reuse_len: int = 0) -> torch.Tensor:
        r"""Cache hidden states into memory."""
        assert mem_len > 0

        if reuse_len is not None and reuse_len > 0:
            output = output[:reuse_len]
        if prev_mem is None:
            new_mem = output[-mem_len:]
        else:
            new_mem = torch.cat([prev_mem, output], dim=0)[-mem_len:]
        return new_mem.detach()

    def _create_causal_attn_mask(self,
                                 seq_len: int,
                                 mem_len: int,
                                 same_length: bool = False) -> torch.Tensor:
        r"""Create causal attention mask of shape
        `(seq_len, mem_len + seq_len)`.
        """
        assert self.r_w_bias is not None
        device = self.r_w_bias.device
        attn_mask = torch.ones(seq_len, seq_len, device=device)
        mask_u = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros(seq_len, mem_len, device=device)
        ret = torch.cat([attn_mask_pad, mask_u], dim=1)
        if same_length:
            mask_l = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :seq_len] + mask_l, ret[:, seq_len:]], 1)
        return ret

    def forward(self,  # type: ignore
                token_ids: torch.LongTensor,
                *args,
                **kwargs) \
            -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        r"""A wrapper for :meth:`_forward`. This layer exists because
        :class:`XLNetDecoder` compute embeddings in the decoder helper.
        Please refer to :meth:`_forward` for the full list of arguments.

        Args:
            token_ids: Shape `[batch_size, seq_len]`.
            **kwargs: Remaining arguments to pass to :meth:`_forward`.
        """
        return self._forward(self.word_embed(token_ids), *args, **kwargs)

    def _forward(self,
                 word_embed: torch.Tensor,
                 segment_ids: Optional[torch.LongTensor] = None,
                 input_mask: Optional[torch.Tensor] = None,
                 memory: Optional[List[torch.Tensor]] = None,
                 permute_mask: Optional[torch.Tensor] = None,
                 target_mapping: Optional[torch.Tensor] = None,
                 bi_data: bool = False,
                 clamp_len: Optional[int] = None,
                 cache_len: int = 0,
                 same_length: bool = False,
                 attn_type: str = 'bi',
                 two_stream: bool = False) \
            -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        r"""Compute XLNet representations for the input.

        Args:
            word_embed: Shape `[batch_size, seq_len, word_embed_dim]`.
            segment_ids: Shape `[batch_size, seq_len]`.
            input_mask: Float tensor of shape `[batch_size, seq_len]`. Note that
                positions with value 1 are masked out.
            memory: Memory from previous batches. A list of length `num_layers`,
                each tensor of shape `[batch_size, mem_len, hidden_dim]`.
            permute_mask: The permutation mask. Float tensor of shape
                `[batch_size, seq_len, seq_len]`.
                A value of 0 for ``permute_mask[i, j, k]`` indicates that
                position `i` attends to position `j` in batch `k`.
            target_mapping: The target token mapping. Float tensor of shape
                `[batch_size, num_targets, seq_len]`.
                A value of 1 for ``target_mapping[i, j, k]`` indicates that
                the `i`-th target token (in order of permutation) in batch `k`
                is the token at position `j`.
                Each row ``target_mapping[i, :, k]`` can have no more than one
                value of 1.
            bi_data (bool): Whether to use bidirectional data input pipeline.
            clamp_len (int): Clamp all relative distances larger than
                :attr:`clamp_len`. A value of -1 means no clamping.
            cache_len (int): Length of memory (number of tokens) to cache.
            same_length (bool): Whether to use the same attention length for
                each token.
            attn_type (str): Attention type. Supported values are `"uni"`
                and `"bi"`.
            two_stream (bool): Whether to use two-stream attention. Only set to
                `True` when pre-training or generating text. Defaults to
                `False`.

        :returns: A tuple of `(output, new_memory)`:

            - **`output`**: The final layer output representations. Shape
              `[batch_size, seq_len, hidden_dim]`.
            - **`new_memory`**: The memory of the current batch.
              If `cache_len` is 0, then `new_memory` is `None`. Otherwise, it is
              a list of length `num_layers`, each tensor of shape
              `[batch_size, cache_len, hidden_dim]`.
              This can be used as the :attr:`memory` argument in the next batch.
        """
        # word_embed: [seq_len, batch_size, word_embed_dim]
        word_embed = word_embed.permute(1, 0, 2)
        # segment_ids: [seq_len, batch_size]
        if segment_ids is not None:
            segment_ids = segment_ids.permute(1, 0)
        # input_mask: [seq_len, batch_size]
        if input_mask is not None:
            input_mask = input_mask.permute(1, 0)
        # memory: A list of length num_layers
        # each tensor of shape [mem_len, batch_size, hidden_dim]
        if memory is not None:
            memory = [m.permute(1, 0, 2) for m in memory]
        # permute_mask: [seq_len, seq_len, batch_size]
        if permute_mask is not None:
            permute_mask = permute_mask.permute(1, 2, 0)
        # target_mapping: [num_targets, seq_len, batch_size]
        if target_mapping is not None:
            target_mapping = target_mapping.permute(1, 2, 0)

        seq_len, batch_size = word_embed.size()[:2]
        mem_len = memory[0].size(0) if memory is not None else 0
        tot_len = seq_len + mem_len
        reuse_len = self._hparams.reuse_len

        # Construct masks.
        masks: List[Optional[torch.Tensor]] = []

        # Causal attention mask.
        if attn_type == 'uni':
            causal_mask = self._create_causal_attn_mask(
                seq_len, mem_len, same_length)
            # attn_mask: (seq_len, tot_len, 1, 1)
            causal_mask = causal_mask.unsqueeze(2).unsqueeze(3)
            masks.append(causal_mask)
        elif attn_type == 'bi':
            pass
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        # Data mask: input mask & permutation mask.
        if input_mask is not None:
            input_mask = input_mask.expand(seq_len, -1, -1)
        data_mask = xlnet_utils.sum_tensors([input_mask, permute_mask])
        if data_mask is not None:
            # All positions in memory can be attended to.
            memory_mask = data_mask.new_zeros(seq_len, mem_len, batch_size)
            # data_mask: (seq_len, tot_len, batch_size, 1)
            data_mask = torch.cat([memory_mask, data_mask], dim=1).unsqueeze(3)
            masks.append(data_mask)

        # Exclude the main diagonal (target tokens) from the mask.
        attn_mask = xlnet_utils.sum_tensors(masks)
        if attn_mask is None:
            final_mask = None
        else:
            attn_mask = (attn_mask > 0)
            final_mask = -torch.eye(seq_len, device=attn_mask.device)
            final_mask = torch.cat([
                final_mask.new_zeros(seq_len, mem_len), final_mask], dim=-1)
            final_mask = final_mask.unsqueeze(2).unsqueeze(3)
            # final_mask: (seq_len, tot_len, batch_size, 1)
            final_mask = ((attn_mask.float() + final_mask) > 0)

        # Construct segment embedding.
        if segment_ids is not None:
            concat_segment_ids = torch.cat([
                segment_ids.new_zeros(mem_len, batch_size), segment_ids])
            segment_matrix = (segment_ids.unsqueeze(1) !=
                              concat_segment_ids.unsqueeze(0)).long()
            segment_matrix = F.one_hot(segment_matrix, num_classes=2).float()
        else:
            segment_matrix = None

        pos_embed = self.pos_embed(
            batch_size, seq_len, tot_len, clamp_len, attn_type, bi_data)
        pos_embed = self.dropout(pos_embed)

        states_h = self.dropout(word_embed)
        if two_stream:
            if target_mapping is not None:
                word_embed_q = self.mask_emb.expand(
                    target_mapping.size(0), batch_size, -1)
            else:
                word_embed_q = word_embed
            states_g = self.dropout(word_embed_q)
        else:
            states_g = None
        new_memory = []

        for idx in range(self._hparams.num_layers):
            cur_memory = memory[idx] if memory is not None else None
            if cache_len > 0:
                new_memory.append(self._cache_mem(
                    states_h, cur_memory, cache_len, reuse_len))
            attn_layer: RelativeMultiheadAttention = self.attn_layers[idx]
            states_h, states_g = attn_layer(
                states_h=states_h, states_g=states_g,
                pos_embed=pos_embed, segment_mat=segment_matrix,
                attn_mask_h=final_mask, attn_mask_g=attn_mask,
                target_mapping=target_mapping, memory=cur_memory)
            states_h = self.ff_layers[idx](states_h)
            if states_g is not None:
                states_g = self.ff_layers[idx](states_g)

        output = self.dropout(states_h if states_g is None else states_g)

        # Now output: [seq_len, batch_size, hidden_dim]
        # new_memory: None or A list of length num_layers,
        # each tensor of shape [cache_len, batch_size, hidden_dim]
        output = output.permute(1, 0, 2)
        if new_memory is not None:
            new_memory = [m.permute(1, 0, 2) for m in new_memory]

        if cache_len == 0:
            return output, None

        return output, new_memory
