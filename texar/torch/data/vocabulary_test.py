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
Unit tests for vocabulary related operations.
"""
import tempfile
import unittest

from texar.torch.data import vocabulary


class VocabularyTest(unittest.TestCase):
    """Tests vocabulary related operations.
    """

    def test_make_defaultdict(self):
        """Tests the _make_defaultdict function.
        """
        keys = ['word', '词']
        values = [0, 1]
        default_value = -1

        dict_ = vocabulary._make_defaultdict(keys, values, default_value)

        self.assertEqual(len(dict_), 2)
        self.assertEqual(dict_['word'], 0)
        self.assertEqual(dict_['词'], 1)
        self.assertEqual(dict_['sth_else'], -1)

    def test_vocab_construction(self):
        """Test vocabulary construction.
        """
        vocab_list = ['word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()

        vocab = vocabulary.Vocab(vocab_file.name)

        self.assertEqual(vocab.size, len(vocab_list) + 4)
        self.assertEqual(
            set(vocab.token_to_id_map_py.keys()),
            set(['word', '词'] + vocab.special_tokens))

        # Tests UNK token
        unk_token_id = vocab.map_tokens_to_ids_py(['new'])
        unk_token_text = vocab.map_ids_to_tokens_py(unk_token_id)
        self.assertEqual(unk_token_text[0], vocab.unk_token)


if __name__ == "__main__":
    unittest.main()
