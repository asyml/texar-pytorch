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
Unit tests for pre-trained BERT tokenizer utils.
"""

import unittest

from texar.torch.data.tokenizers.bert_tokenizer_utils import (
    BasicTokenizer, WordpieceTokenizer, _is_control,
    _is_punctuation, _is_whitespace)


class BERTTokenizerUtilsTest(unittest.TestCase):

    def test_chinese(self):

        tokenizer = BasicTokenizer()

        self.assertListEqual(
            tokenizer.tokenize(u"ah\u535A\u63A8zz"),
            [u"ah", u"\u535A", u"\u63A8", u"zz"])

    def test_basic_tokenizer_lower(self):

        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["hello", "!", "how", "are", "you", "?"])

        self.assertListEqual(tokenizer.tokenize(u"H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):

        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["HeLLo", "!", "how", "Are", "yoU", "?"])

    def test_wordpiece_tokenizer(self):

        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])
        self.assertListEqual(tokenizer.tokenize_with_span(""), [])

        self.assertListEqual(
            tokenizer.tokenize("unwanted running"),
            ["un", "##want", "##ed", "runn", "##ing"])
        self.assertListEqual(
            tokenizer.tokenize_with_span("unwanted running"), [
                ('un', 0, 2), ('##want', 2, 6),
                ('##ed', 6, 8), ('runn', 9, 13),
                ('##ing', 13, 16)])

        self.assertListEqual(
            tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])
        self.assertListEqual(
            tokenizer.tokenize_with_span("unwantedX running"), [
                ('[UNK]', 0, 9),
                ('runn', 10, 14),
                 ('##ing', 14, 17)])

    def test_is_whitespace(self):

        self.assertTrue(_is_whitespace(u" "))
        self.assertTrue(_is_whitespace(u"\t"))
        self.assertTrue(_is_whitespace(u"\r"))
        self.assertTrue(_is_whitespace(u"\n"))
        self.assertTrue(_is_whitespace(u"\u00A0"))

        self.assertFalse(_is_whitespace(u"A"))
        self.assertFalse(_is_whitespace(u"-"))

    def test_is_control(self):

        self.assertTrue(_is_control(u"\u0005"))

        self.assertFalse(_is_control(u"A"))
        self.assertFalse(_is_control(u" "))
        self.assertFalse(_is_control(u"\t"))
        self.assertFalse(_is_control(u"\r"))

    def test_is_punctuation(self):

        self.assertTrue(_is_punctuation(u"-"))
        self.assertTrue(_is_punctuation(u"$"))
        self.assertTrue(_is_punctuation(u"`"))
        self.assertTrue(_is_punctuation(u"."))

        self.assertFalse(_is_punctuation(u"A"))
        self.assertFalse(_is_punctuation(u" "))


if __name__ == "__main__":
    unittest.main()
