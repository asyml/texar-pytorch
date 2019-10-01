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
XLNet encoder.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from texar.torch.modules.encoders.encoder_base import EncoderBase
from texar.torch.modules.pretrained.xlnet import PretrainedXLNetMixin
from texar.torch.modules.pretrained.xlnet_utils import (
    PositionWiseFF, RelativeMultiheadAttention, RelativePositionalEncoding,
    params_except_in)
from texar.torch.utils.utils import dict_fetch, sum_tensors

__all__ = [
    "XLNetEncoder",
]


class XLNetEncoder(EncoderBase, PretrainedXLNetMixin):
    r"""Raw XLNet module for encoding sequences. Please see
    :class:`~texar.torch.modules.PretrainedXLNetMixin` for a brief description
    of XLNet.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``xlnet-based-cased``). Please refer to
            :class:`~texar.torch.modules.PretrainedXLNetMixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """
    _IS_DECODE = False

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        super().__init__(hparams=hparams)
        self.load_pretrained_config(pretrained_model_name, cache_dir)

        num_layers = self._hparams.num_layers
        num_heads = self._hparams.num_heads
        head_dim = self._hparams.head_dim

        self.word_embed = nn.Embedding(self._hparams.vocab_size,
                                       self._hparams.hidden_dim)
        self.pos_embed = RelativePositionalEncoding(
            hparams={
                "dim": self._hparams.hidden_dim,
                "max_seq_len": self._hparams.max_seq_length,
            })
        self.dropout = nn.Dropout(self._hparams.dropout)

        self.r_r_bias = None
        self.r_w_bias = None
        self.r_s_bias = None

        if not self._hparams.untie_r:
            self.r_r_bias = nn.Parameter(torch.Tensor(num_heads, head_dim))
            self.r_w_bias = nn.Parameter(torch.Tensor(num_heads, head_dim))
            self.r_s_bias = (nn.Parameter(torch.Tensor(num_heads, head_dim))
                             if self._hparams.use_segments else None)

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

        if self._IS_DECODE:
            self.lm_bias = nn.Parameter(torch.zeros(self._hparams.vocab_size))

        self.init_pretrained_weights()

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
                "max_seq_length": 512,
                "initializer": None,
                "name": "xlnet_encoder",
            }

        Here:

        The default parameters are values for cased XLNet-Base model.

        `"pretrained_model_name"`: str or None
            The name of the pre-trained XLNet model. If None, the model
            will be randomly initialized.

        `"untie_r"`: bool
            Whether to untie the biases in attention.

        `"num_layers"`: int
            The number of stacked layers.

        `"mem_len"`: int
            The number of tokens to cache.

        `"reuse_len"`: int
            The number of tokens in the current batch to be cached and reused
            in the future.

        `"num_heads"`: int
            The number of attention heads.

        `"hidden_dim"`: int
            The hidden size.

        `"head_dim"`: int
            The dimension size of each attention head.

        `"dropout"`: float
            Dropout rate.

        `"attention_dropout"`: float
            Dropout rate on attention probabilities.

        `"use_segments"`: bool
            Whether to use segment embedding.

        `"ffn_inner_dim"`: int
            The hidden size in feed-forward layers.

        `"activation"`: str
            `relu` or `gelu`.

        `"vocab_size"`: int
            The vocabulary size.

        `"max_seq_length"`: int
            The maximum sequence length for `RelativePositionalEncoding`.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.torch.core.get_initializer` for details.

        `"name"`: str
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
            'max_seq_length': 512,
            'initializer': None,
            'name': "xlnet_encoder",
            '@no_typecheck': ['pretrained_model_name'],
        }

    def param_groups(self,
                     lr: Optional[float] = None,
                     lr_layer_scale: float = 1.0,
                     decay_base_params: bool = False):
        r"""Create parameter groups for optimizers. When
        :attr:`lr_layer_decay_rate` is not 1.0, parameters from each layer form
        separate groups with different base learning rates.

        The return value of this method can be used in the constructor of
        optimizers, for example:

        .. code-block:: python

            model = XLNetEncoder(...)
            param_groups = model.param_groups(lr=2e-5, lr_layer_scale=0.8)
            optim = torch.optim.Adam(param_groups)

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

            num_layers = self._hparams.num_layers
            base_group = {
                "params": params_except_in(
                    self, ['attn_layers', 'ff_layers']),
                "lr": lr * (lr_layer_scale ** num_layers
                            if decay_base_params else 1.0)
            }
            param_groups = [base_group]
            for idx in range(num_layers):
                decay_rate = lr_layer_scale ** (num_layers - idx - 1)
                param_group = {
                    "params": [*self.attn_layers[idx].parameters(),
                               *self.ff_layers[idx].parameters()],
                    "lr": lr * decay_rate,
                }
                param_groups.append(param_group)
            return param_groups
        return self.parameters()

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output.
        """
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
                inputs: Union[torch.Tensor, torch.LongTensor],
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
            inputs: Either a **2D Tensor** of shape `[batch_size, max_time]`,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
            segment_ids: Shape `[batch_size, max_time]`.
            input_mask: Float tensor of shape `[batch_size, max_time]`. Note
                that positions with value 1 are masked out.
            memory: Memory from previous batches. A list of length `num_layers`,
                each tensor of shape `[batch_size, mem_len, hidden_dim]`.
            permute_mask: The permutation mask. Float tensor of shape
                `[batch_size, max_time, max_time]`.
                A value of 0 for ``permute_mask[i, j, k]`` indicates that
                position `i` attends to position `j` in batch `k`.
            target_mapping: The target token mapping. Float tensor of shape
                `[batch_size, num_targets, max_time]`.
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
              `[batch_size, max_time, hidden_dim]`.
            - **`new_memory`**: The memory of the current batch.
              If `cache_len` is 0, then `new_memory` is `None`. Otherwise, it is
              a list of length `num_layers`, each tensor of shape
              `[batch_size, cache_len, hidden_dim]`.
              This can be used as the :attr:`memory` argument in the next batch.
        """
        if inputs.dim() == 2:
            word_embeds = self.word_embed(inputs)
        elif inputs.dim() == 3:
            word_embeds = torch.tensordot(inputs, self.word_embed.weight,
                                          dims=([-1], [0]))
        else:
            raise ValueError("'inputs' should be a 2D or 3D tensor.")

        return self._forward(word_embed=word_embeds,
                             segment_ids=segment_ids,
                             input_mask=input_mask,
                             memory=memory,
                             permute_mask=permute_mask,
                             target_mapping=target_mapping,
                             bi_data=bi_data,
                             clamp_len=clamp_len,
                             cache_len=cache_len,
                             same_length=same_length,
                             attn_type=attn_type,
                             two_stream=two_stream)

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
        r"""Compute XLNet representations for the input. This layer exists
        because :class:`XLNetDecoder` compute embeddings in the decoder helper.
        `word_embed` has shape `[batch_size, max_time, word_embed_dim]`.
        Please refer to :meth:`forward` for the detailed information of other
        arguments.
        """
        # seq_len == max_time
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
        data_mask = sum_tensors([input_mask, permute_mask])
        if data_mask is not None:
            # All positions in memory can be attended to.
            memory_mask = data_mask.new_zeros(seq_len, mem_len, batch_size)
            # data_mask: (seq_len, tot_len, batch_size, 1)
            data_mask = torch.cat([memory_mask, data_mask], dim=1).unsqueeze(3)
            masks.append(data_mask)

        # Exclude the main diagonal (target tokens) from the mask.
        attn_mask = sum_tensors(masks)
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
        states_g = None
        if two_stream:
            if target_mapping is not None:
                word_embed_q = self.mask_emb.expand(
                    target_mapping.size(0), batch_size, -1)
            else:
                word_embed_q = word_embed
            states_g = self.dropout(word_embed_q)
        new_memory = []

        for idx in range(self._hparams.num_layers):
            cur_memory = memory[idx] if memory is not None else None
            if cache_len > 0:
                new_memory.append(self._cache_mem(
                    states_h, cur_memory, cache_len, reuse_len))
            attn_layer: RelativeMultiheadAttention
            attn_layer = self.attn_layers[idx]  # type: ignore
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
