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
Utility for reading important params from T5 gin files.
"""

import ast

IMPORTANT_PARAMS = ('d_ff',
                    'd_kv',
                    'd_model',
                    'dropout',
                    'num_heads',
                    'num_layers',
                    'inputs_length'
                    )


def read_t5_gin_config_file(config_file_path):
    r"""Simple helper function to read a gin file
    and get hyperparameters for T5

    :return:
    """
    config = {}

    with open(config_file_path, 'r') as gin_file:
        for line in gin_file:
            if line.startswith(IMPORTANT_PARAMS):
                assignment = line.strip().split()
                assert len(assignment) == 3
                arg_name, _, value = assignment
                config[arg_name] = ast.literal_eval(value)

    return config

