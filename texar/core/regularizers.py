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
Regularizers
"""

import torch

__all__ = [
    'Regularizer',
    'L1L2',
    'l1',
    'l2',
    'l1_l2'
]


class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    Args:
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):  # pylint: disable=redefined-outer-name
        self.l1 = float(l1)
        self.l2 = float(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += torch.sum(self.l1 * torch.abs(x))
        if self.l2:
            regularization += torch.sum(self.l2 * torch.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}


# Aliases.
def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):  # pylint: disable=redefined-outer-name
    return L1L2(l1=l1, l2=l2)
