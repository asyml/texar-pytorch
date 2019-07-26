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
Utils of Pre-trained Modules.
"""

import os
import sys
from pathlib import Path

__all__ = [
    "default_download_dir",
]


def default_download_dir(name: str) -> Path:
    r"""Return the directory to which packages will be downloaded by default.
    """
    package_dir: Path = Path(__file__).parents[3]  # 4 levels up
    if os.access(package_dir, os.W_OK):
        texar_download_dir = package_dir / 'texar_download'
    else:
        if sys.platform == 'win32' and 'APPDATA' in os.environ:
            # On Windows, use %APPDATA%
            home_dir = Path(os.environ['APPDATA'])
        else:
            # Otherwise, install in the user's home directory.
            home_dir = Path.home()

        texar_download_dir = home_dir / 'texar_download'

    if not texar_download_dir.exists():
        texar_download_dir.mkdir(parents=True)

    return texar_download_dir / name
