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
Utils for unit tests.
"""

import os
import unittest

pretrained_test = unittest.skipUnless(
    os.environ.get('TEST_PRETRAINED', 0) or os.environ.get('TEST_ALL', 0),
    "Test requires loading pre-trained checkpoints. "
    "Set `TEST_PRETRAINED=1` or `TEST_ALL=1` to run.")
data_test = unittest.skipUnless(
    os.environ.get('TEST_DATA', 0) or os.environ.get('TEST_ALL', 0),
    "Test requires loading large data files. "
    "Set `TEST_DATA=1` or `TEST_ALL=1` to run.")
