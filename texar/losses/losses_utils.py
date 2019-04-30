# Copyright 2018 The Texar Authors. All Rights Reserved.
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
Various utilities for losses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch




def reduce_dimensions(tensor, average_axes=None, sum_axes=None, keepdims=None):
    """Average or sum over dimensions of :attr:`tensor`.

    :attr:`average_axes` and :attr:`sum_axes` must be mutually exclusive. That
    is, elements in `average_axes` must not be contained in
    `sum_axes`, and vice versa.

    Args:
        tensor: A tensor to reduce.
        average_axes (optional): A (list of) `int` that indicates the
            dimensions to reduce by taking average.
        sum_axes (optional): A (list of) `int` that indicates the
            dimensions to reduce by taking sum.
        keepdims (optional): If `True`, retains reduced dimensions with
            length 1.
    """
    reduced_axes = set()
    if average_axes is not None:
        if not isinstance(average_axes, (list, tuple)):
            average_axes = [average_axes]
        if len(average_axes) > 0:
            for average_axis in average_axes:
                tensor = torch.mean(tensor, dim=average_axis, keepdim=True)
            reduced_axes.update(average_axes)

    if sum_axes is not None:
        if not isinstance(sum_axes, (list, tuple)):
            sum_axes = [sum_axes]
        if len(sum_axes) > 0:
            for sum_axis in sum_axes:
                tensor = torch.sum(tensor, dim=sum_axis, keepdim=True)
            reduced_axes.update(sum_axes)

            if average_axes is not None:
                if len(reduced_axes) != len(average_axes) + len(sum_axes):
                    raise ValueError('`average_axes` and `sum_axes` must not '
                                     'have overlapped elements.')
    if not keepdims:
        tensor = torch.squeeze(tensor, dim=list(reduced_axes))

    return tensor




