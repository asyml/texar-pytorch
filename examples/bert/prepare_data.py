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
"""Produces TFRecord files and modifies data configuration file
"""

import argparse
import logging
import os

import texar as tx
import utils.data_utils as data_utils
import utils.tokenization as tokenization

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task", type=str, default="MRPC",
    choices=['COLA', 'MNLI', 'MRPC', 'XNLI', 'SST'],
    help="The task to run experiment on.")
parser.add_argument(
    "--vocab_file", type=str,
    default='bert_pretrained_models/uncased_L-12_H-768_A-12/vocab.txt',
    help="The one-wordpiece-per-line vocabary file directory.")
parser.add_argument(
    "--max_seq_length", type=int, default=128,
    help="The maxium length of sequence, longer sequence will be trimmed.")
parser.add_argument(
    "--tfrecord_output_dir", type=str, default=None,
    help="The output directory where the TFRecord files will be generated. "
         "By default it will be set to 'data/{task}'. E.g.: if "
         "task is 'MRPC', it will be set as 'data/MRPC'")
parser.add_argument(
    "--do_lower_case", type=bool, default=True,
    help="Whether to lower case the input text. Should be True for uncased "
         "models and False for cased models.")
args = parser.parse_args()

logging.root.setLevel(logging.INFO)


def prepare_data():
    """
    Builds the model and runs.
    """
    # Loads data
    logging.info("Loading data")

    task_datasets_rename = {
        "COLA": "CoLA",
        "SST": "SST-2",
    }

    data_dir = f'data/{args.task}'
    if args.task.upper() in task_datasets_rename:
        data_dir = f'data/{task_datasets_rename[args.task]}'

    if args.tfrecord_output_dir is None:
        tfrecord_output_dir = data_dir
    else:
        tfrecord_output_dir = args.tfrecord_output_dir
    tx.utils.maybe_create_dir(tfrecord_output_dir)

    processors = {
        "COLA": data_utils.ColaProcessor,
        "MNLI": data_utils.MnliProcessor,
        "MRPC": data_utils.MrpcProcessor,
        "XNLI": data_utils.XnliProcessor,
        'SST': data_utils.SSTProcessor
    }
    processor = processors[args.task]()

    from config_data import feature_original_types

    num_classes = len(processor.get_labels())
    num_train_data = len(processor.get_train_examples(data_dir))
    logging.info(f'num_classes:{num_classes}; num_train_data:{num_train_data}')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file,
        do_lower_case=args.do_lower_case)

    # Produces TFRecord files
    data_utils.prepare_record_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_seq_length=args.max_seq_length,
        output_dir=tfrecord_output_dir,
        feature_original_types=feature_original_types)
    modify_config_data(args.max_seq_length, num_train_data, num_classes)


def modify_config_data(max_seq_length, num_train_data, num_classes):
    # Modify the data configuration file
    config_data_exists = os.path.isfile('./config_data.py')
    if config_data_exists:
        with open("./config_data.py", 'r') as file:
            filedata = file.read()
            filedata_lines = filedata.split('\n')
            idx = 0
            while True:
                if idx >= len(filedata_lines):
                    break
                line = filedata_lines[idx]
                if (line.startswith('num_classes =') or
                        line.startswith('num_train_data =') or
                        line.startswith('max_seq_length =')):
                    filedata_lines.pop(idx)
                    idx -= 1
                idx += 1

            if len(filedata_lines) > 0:
                insert_idx = 1
            else:
                insert_idx = 0
            filedata_lines.insert(
                insert_idx, f'{"num_train_data"} = {num_train_data}')
            filedata_lines.insert(
                insert_idx, f'{"num_classes"} = {num_classes}')
            filedata_lines.insert(
                insert_idx, f'{"max_seq_length"} = {max_seq_length}')

        with open("./config_data.py", 'w') as file:
            file.write('\n'.join(filedata_lines))
        logging.info("config_data.py has been updated")
    else:
        logging.info("config_data.py cannot be found")

    logging.info("Data preparation finished")


def main():
    """ Starts the data preparation
    """
    prepare_data()


if __name__ == "__main__":
    main()
