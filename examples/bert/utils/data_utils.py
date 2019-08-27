# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is the Data Loading Pipeline for Sentence Classifier Task adapted from:
    `https://github.com/google-research/bert/blob/master/run_classifier.py`
"""

import csv
import logging
import os

import texar.torch as tx
from utils import tokenization


class InputExample:
    r"""A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        r"""Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    r"""A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor:
    r"""Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        r"""Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        r"""Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        r"""Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        r"""Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        r"""Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines


class SSTProcessor(DataProcessor):
    r"""Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        r"""See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        r"""Creates examples for the training and dev sets."""
        examples = []
        if set_type in ('train', 'dev'):
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = f"{set_type}-{i}"
                text_a = tokenization.convert_to_unicode(line[0])
                # Single sentence classification, text_b doesn't exist
                text_b = None
                label = tokenization.convert_to_unicode(line[1])
                examples.append(InputExample(guid=guid, text_a=text_a,
                                             text_b=text_b, label=label))
        if set_type == 'test':
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = f"{set_type}-{i}"
                text_a = tokenization.convert_to_unicode(line[1])
                # Single sentence classification, text_b doesn't exist
                text_b = None
                label = '0'  # arbitrary set as 0
                examples.append(InputExample(guid=guid, text_a=text_a,
                                             text_b=text_b, label=label))
        return examples


class XnliProcessor(DataProcessor):
    r"""Processor for the XNLI data set."""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        r"""See base class."""
        lines = self._read_tsv(
            os.path.join(data_dir, "multinli",
                         f"multinli.train.{self.language}.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"train-{i}"
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            if label == tokenization.convert_to_unicode("contradictory"):
                label = tokenization.convert_to_unicode("contradiction")
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        r"""See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"dev-{i}"
            language = tokenization.convert_to_unicode(line[0])
            if language != tokenization.convert_to_unicode(self.language):
                continue
            text_a = tokenization.convert_to_unicode(line[6])
            text_b = tokenization.convert_to_unicode(line[7])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_labels(self):
        r"""See base class."""
        return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
    r"""Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")),
            "test")

    def get_labels(self):
        r"""See base class."""
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines, set_type):
        r"""Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{tokenization.convert_to_unicode(line[0])}"
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            if set_type == "test":
                label = "contradiction"
            else:
                label = tokenization.convert_to_unicode(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    r"""Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train")

    def get_dev_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        r"""See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        r"""Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    r"""Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train")

    def get_dev_examples(self, data_dir):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        r"""See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        r"""Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = f"{set_type}-{i}"
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[3])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=None, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    r"""Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention rule is:
    # (a) For sequence pairs:
    #   tokens: [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #    segment_ids: 0 0 0 0 0 0 0 0                       1 1 1 1 1 1
    # (b) For single sequences:
    #   tokens: [CLS] the dog is hairy . [SEP]
    #   sigment_ids: 0 0 0 0 0 0 0
    #
    # Where "segment_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    # here we disable the verbose printing of the data
    if ex_index < 0:
        logging.info("*** Example ***")
        logging.info("guid: %s", example.guid)
        logging.info("tokens: %s", " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("input_ids length: %d", len(input_ids))
        logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logging.info("label: %s (id = %d)", example.label, label_id)

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file,
        feature_original_types):
    r"""Convert a set of `InputExample`s to a pickled file."""

    with tx.data.RecordData.writer(
            output_file, feature_original_types) as writer:
        for (ex_index, example) in enumerate(examples):
            feature = convert_single_example(ex_index, example, label_list,
                                             max_seq_length, tokenizer)

            features = {
                "input_ids": feature.input_ids,
                "input_mask": feature.input_mask,
                "segment_ids": feature.segment_ids,
                "label_ids": feature.label_id
            }
            writer.write(features)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    r"""Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then
    # each token that's truncated likely contains more information than a
    # longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def prepare_record_data(processor, tokenizer,
                        data_dir, max_seq_length, output_dir,
                        feature_original_types):
    r"""Prepare record data.
    Args:
        processor: Data Preprocessor, which must have get_labels,
            get_train/dev/test/examples methods defined.
        tokenizer: The Sentence Tokenizer. Generally should be
            SentencePiece Model.
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the pickled file in.
        feature_original_types: The original type of the feature.
    """
    label_list = processor.get_labels()

    train_examples = processor.get_train_examples(data_dir)
    train_file = os.path.join(output_dir, "train.pkl")
    file_based_convert_examples_to_features(
        train_examples, label_list, max_seq_length,
        tokenizer, train_file, feature_original_types)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_file = os.path.join(output_dir, "eval.pkl")
    file_based_convert_examples_to_features(
        eval_examples, label_list,
        max_seq_length, tokenizer, eval_file, feature_original_types)

    test_examples = processor.get_test_examples(data_dir)
    test_file = os.path.join(output_dir, "predict.pkl")
    file_based_convert_examples_to_features(
        test_examples, label_list,
        max_seq_length, tokenizer, test_file, feature_original_types)
