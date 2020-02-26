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
Unit tests for the utils of pre-trained ELMo tokenizer.

Code adapted from:
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/data/token_indexers/elmo_indexer_test.py`
"""

import unittest

from texar.torch.data.tokenizers.elmo_tokenizer_utils import (
    ELMoCharacterMapper, batch_to_ids)


class ELMoTokenizerUtilsTest(unittest.TestCase):

    def test_bos_to_char_ids(self):
        mapper = ELMoCharacterMapper()
        indices = mapper.convert_word_to_char_ids('<S>')
        # [<begin word>, <begin sentence>, <end word>, <padding>, ... <padding>]
        expected_indices = [259, 257, 260]
        expected_indices.extend([261] * (50 - len(expected_indices)))
        self.assertEqual(indices, expected_indices)

    def test_eos_to_char_ids(self):
        mapper = ELMoCharacterMapper()
        indices = mapper.convert_word_to_char_ids('</S>')
        expected_indices = [259, 258, 260]
        expected_indices.extend([261] * (50 - len(expected_indices)))
        self.assertEqual(indices, expected_indices)

    def test_unicode_to_char_ids(self):
        mapper = ELMoCharacterMapper()
        indices = mapper.convert_word_to_char_ids(chr(256) + "t")
        expected_indices = [259, 197, 129, 117, 260]
        expected_indices.extend([261] * (50 - len(expected_indices)))
        self.assertEqual(indices, expected_indices)

    def test_additional_tokens(self):
        mapper = ELMoCharacterMapper(tokens_to_add={"<first>": 1})
        indices = mapper.convert_word_to_char_ids("<first>")
        expected_indices = [259, 2, 260]
        expected_indices.extend([261] * (50 - len(expected_indices)))
        self.assertEqual(indices, expected_indices)

    def test_batch_to_ids(self):
        sentences = [['First', 'sentence', '.'], ['Another', '.']]
        indices = batch_to_ids(sentences)
        expected_indices = [[
            [259,  71, 106, 115, 116, 117, 260] + [261] * 43,
            [259, 116, 102, 111, 117, 102, 111, 100, 102, 260] + [261] * 40,
            [259,  47, 260] + [261] * 47], [
            [259, 66, 111, 112, 117, 105, 102, 115, 260] + [261] * 41,
            [259, 47, 260] + [261] * 47,
            [0] * 50]]
        self.assertEqual(indices.tolist(), expected_indices)


if __name__ == "__main__":
    unittest.main()
