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
Score function.
"""
import subprocess


def scores(path):
    bashCommand = 'perl conlleval'
    process = subprocess.Popen(
        bashCommand.split(), stdout=subprocess.PIPE, stdin=open(path))
    output, _ = process.communicate()
    output = output.decode().split('\n')[1].split('%; ')
    output = [out.split(' ')[-1] for out in output]
    acc, prec, recall, fb1 = tuple(output)
    return float(acc), float(prec), float(recall), float(fb1)
