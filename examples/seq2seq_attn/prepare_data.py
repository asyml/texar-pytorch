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
"""Downloads data.
"""

import argparse

import texar.torch as tx

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', type=str, default="iwslt14",
    help="Data to download [iwslt14|toy_copy]")
args = parser.parse_args()


def main() -> None:
    """Downloads data.
    """
    if args.data == 'iwslt14':
        tx.data.maybe_download(
            urls='https://drive.google.com/file/d/'
                 '1y4mUWXRS2KstgHopCS9koZ42ENOh6Yb9/view?usp=sharing',
            path='./',
            filenames='iwslt14.zip',
            extract=True)
    elif args.data == 'toy_copy':
        tx.data.maybe_download(
            urls='https://drive.google.com/file/d/'
                 '1fENE2rakm8vJ8d3voWBgW4hGlS6-KORW/view?usp=sharing',
            path='./',
            filenames='toy_copy.zip',
            extract=True)
    else:
        raise ValueError(f'Unknown dataset: {args.data}')


if __name__ == '__main__':
    main()
