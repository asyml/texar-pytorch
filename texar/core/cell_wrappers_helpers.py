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
Various helper classes and utilities for cell wrappers.
"""

# pylint: disable=too-many-arguments, too-many-instance-attributes
# pylint: disable=missing-docstring  # does not support generic classes

import torch


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    if isinstance(attention_mechanism, _BaseAttentionMechanismV2):
        alignments, next_attention_state = attention_mechanism(
            [cell_output, attention_state])
    else:
        # For other class, assume they are following _BaseAttentionMechanism,
        # which takes query and state as separate parameter.
        alignments, next_attention_state = attention_mechanism(
            cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = torch.unsqueeze(alignments, dim=1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context_ = torch.matmul(expanded_alignments, attention_mechanism.values)
    context_ = torch.squeeze(context_, dim=1)

    if attention_layer is not None:
        attention = attention_layer(torch.cat((cell_output, context_), dim=1))
    else:
        attention = context_

    return attention, alignments, next_attention_state











