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
"""Utils of pre-trained GPT2 tokenizer.
"""

from functools import lru_cache

__all__ = [
    "bytes_to_unicode",
    "get_pairs",
]


@lru_cache()
def bytes_to_unicode():
    r"""Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings. This means you need a
    large number of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing
    around 5K for decent coverage. This is a significant percentage of your
    normal, say, 32K bpe vocab. To avoid that, we want lookup tables between
    utf-8 bytes and unicode strings. And avoids mapping to whitespace/control
    characters the bpe code barfs on.
    """
    _chr = chr
    bs = list(range(ord("!"), ord("~") + 1)) + list(
        range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    r"""Return set of symbol pairs in a word. Word is represented as tuple of
    symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
