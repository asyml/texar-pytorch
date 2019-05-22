# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2018 Texar
# ==============================================================================
"""
This script is adapted from the tensor2tensor repository.
"""

#import tensorflow as tf
import torch

# pylint: disable=invalid-name, too-many-arguments, too-many-locals

class PadRemover(object):
    """Helper to remove padding from a tensor before sending to the experts.
    The padding is computed for one reference tensor containing the padding mask
    and then can be applied to any other tensor of shape [dim_origin,...].
    Example::
            input = [
                [tok1, tok2],
                [tok3, tok4],
                [0, 0],
                [0, 0],
                [tok5, tok6],
                [0, 0],
            ]
            output = [
                [tok1, tok2],
                [tok3, tok4],
                [tok5, tok6],
            ]
    """

    def __init__(self, pad_mask: torch.Tensor):
        """Compute and store the location of the padding.
        Args:
            pad_mask (tf.Tensor): Reference padding tensor of shape
                [batch_size,length] or [dim_origin]
                (dim_origin=batch_size*length)
                containing non-zeros positive values to indicate padding
                location.
        """
        self.nonpad_ids = None
        self.dim_origin = None

        #with tf.name_scope("pad_reduce/get_ids"):
        pad_mask = torch.reshape(pad_mask, [-1])    # Flatten the batch
        # nonpad_ids contains coordinates of zeros rows (as pad_mask is
        # float32, checking zero equality is done with |x| < epsilon, with
        # epsilon=1e-9 as standard, here pad_mask only contains positive
        # values so tf.abs would be redundant)
        ones_mask = torch.ones_like(pad_mask)
        zeros_mask = torch.zeros_like(pad_mask)
        non_pad = torch.where(pad_mask < 1e-9, ones_mask, zeros_mask)
        self.nonpad_ids = torch.nonzero(non_pad).squeeze()
        self.dim_origin = pad_mask.size()[:1]

    def remove(self, x: torch.Tensor):
        """Remove padding from the given tensor.
        Args:
            x: A Tensor of shape [dim_origin,...]
        Returns:
            A tensor of shape [dim_compressed,...] with dim_compressed
            <= dim_origin
        """
        #with tf.name_scope("pad_reduce/remove"):
        #x_shape = list(x.size())
        '''x = tf.gather_nd(
            x,
            indices=self.nonpad_ids,
        )'''
        x = x.index_select(0, self.nonpad_ids)
        #if not context.in_eager_mode():
        # This is a hack but for some reason, gather_nd return a tensor of
        # undefined shape, so the shape is set up manually
        #x.set_shape([None] + x_shape[1:])
        return x

    def restore(self, x):
        """Add padding back to the given tensor.
        Args:
            x: A Tensor of shape [dim_compressed,...]
        Returns:
            A tensor of shape [dim_origin,...] with
            dim_compressed >= dim_origin. The
            dim is restored from the original reference tensor
        """

        shape = self.dim_origin + x.size()[1:]
        space = torch.zeros(shape)

        x = space.scatter_(0, self.nonpad_ids.unsqueeze(1).expand(x.size()), x)
        return x