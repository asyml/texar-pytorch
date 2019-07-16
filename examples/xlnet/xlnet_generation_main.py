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
"""Example of building XLNet language model for sample generation.
"""

import argparse
import torch
import sentencepiece as spm

import xlnet
import xlnet.model.decoder
from texar.modules import TopKSampleEmbeddingHelper, TopPSampleEmbeddingHelper

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None,
                    help="Checkpoint to load model weights from. Use "
                         "`--pretrain_checkpoint` instead if loading XLNet "
                         "pretrained checkpoint.")
parser.add_argument('--pretrain_checkpoint', type=str,
                    default="pretrained/xlnet_cased_L-24_H-1024_A-16/"
                            "xlnet_model.ckpt",
                    help="XLNet pretrained model checkpoint. Ignored if "
                         "'--checkpoint' is specified.")
parser.add_argument('--pretrain_model_dir', type=str,
                    default="pretrained/xlnet_cased_L-24_H-1024_A-16",
                    help="The directory of pretrained model, for loading "
                         "vocabulary, etc.")
parser.add_argument('--seed', type=int, default=None, help="Random seed.")
parser.add_argument('--nsamples', type=int, default=1,
                    help="Total number of samples to generate. Used in "
                         "non-interactive mode.")
parser.add_argument('--batch_size', type=int, default=1,
                    help="The batch size of input.")
parser.add_argument('--max_decoding_length', type=int, default=100,
                    help="The maximun length of generated text.")
parser.add_argument('--temperature', type=float, default=0.7,
                    help="Softmax temperature for top-k sample decoding. Must "
                         "be strictly greater than 0. Defaults to 0.7.")
parser.add_argument('--top_k', type=int, default=40,
                    help="The number of top most likely candidates to choose "
                         "from at each step. This is use "
                         "TopKSampleEmbeddingHelper for decoding. Ignored if "
                         "'p' is given.")
parser.add_argument('--top_p', type=float, default=None,
                    help="Select tokens with cumulative probability of at most "
                         "'top_p' when arranged in decreasing order. This "
                         "will use TopPSampleEmbeddingHelper for decoding.")
parser.add_argument('--is_interactive', action='store_true',
                    help="Interactive mode or not.")
parser.add_argument('--sentence_piece', type=str,
                    default="pretrained/xlnet_cased_L-24_H-1024_A-16/"
                            "spiece.model",
                    help="Location to load sentence piece model from.")

args = parser.parse_args()


def main():

    pretrain_checkpoint = args.pretrain_checkpoint
    sentence_piece_model = args.sentence_piece
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = 'cpu'

    model = xlnet.model.decoder.XLNetDecoder()
    xlnet.model.load_from_tf_checkpoint(model, pretrain_checkpoint)
    model = model.to(device)
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(sentence_piece_model)
    tokenize_fn = xlnet.data.create_tokenize_fn(sp_model, False)

    # A lengthy padding text used to workaround lack of context for short
    # prompts. Refer to https://github.com/rusiaaman/XLNet-gen for the rationale
    # behind this.
    pad_txt = """
        Texar-PyTorch is an open-source toolkit based on PyTorch, aiming to
        support a broad set of machine learning, especially text generation
        tasks, such as machine translation, dialog, summarization, content
        manipulation, language modeling, and so on. Texar is designed for both
        researchers and practitioners for fast prototyping and
        experimentation.
        With the design goals of modularity, versatility, and extensibility in
        mind, Texar extracts the common patterns underlying the diverse tasks
        and methodologies, creates a library of highly reusable modules and
        functionalities, and facilitates arbitrary model architectures and
        algorithmic paradigms. """
    pad_ids = tokenize_fn(pad_txt)
    pad_ids.append(xlnet.data.utils.EOD_ID)

    def split_by(xs, y):
        p = 0
        for idx, x in enumerate(xs):
            if x == y:
                if idx - p > 0:
                    yield xs[p:idx]
                p = idx + 1
        if len(xs) - p > 0:
            yield xs[p:]

    @torch.no_grad()
    def sample(text: str, length: int = 100, n_samples=3, **kwargs):
        model.eval()
        text = text.replace("\n", "<eop>")
        tokens = pad_ids + tokenize_fn(text)
        tokens = torch.tensor(tokens, device=device).expand(n_samples, -1)
        if args.top_p:
            kwargs["p"] = args.top_p
            decode_output, _ = model(tokens, max_decoding_length=length,
                                     print_steps=True,
                                     helper_type=TopPSampleEmbeddingHelper,
                                     **kwargs)
        else:
            kwargs["top_k"] = args.top_k
            decode_output, _ = model(tokens, max_decoding_length=length,
                                     print_steps=True,
                                     helper_type=TopKSampleEmbeddingHelper,
                                     **kwargs)
        decode_samples = decode_output.sample_id.tolist()
        for idx, sample_tokens in enumerate(decode_samples):
            print(f"=== Sample {idx} ===")
            output = "\n".join(sp_model.DecodeIds(xs) for xs in split_by(
                sample_tokens, xlnet.data.utils.special_symbols["<eop>"]))
            print(output)

    nsamples = args.nsamples
    batch_size = args.batch_size
    max_decoding_length = args.max_decoding_length
    assert nsamples % batch_size == 0, (
        "nsamples must be dividable by batch_size")

    if args.is_interactive:
        while True:

            try:
                raw_text = input("Model input >>> ")
                while not raw_text:
                    print('Input should not be empty!')
                    raw_text = input("Model input >>> ")
                sample(text=raw_text, length=max_decoding_length,
                       n_samples=batch_size)
            except EOFError:
                print("EOF entered, quitting.")
                exit(0)
    else:
        # Generate samples from scratch
        for _ in range(nsamples // batch_size):
            for _ in range(args.batch_size):
                sample(text="<BOS>", length=max_decoding_length,
                       n_samples=batch_size)


if __name__ == '__main__':
    main()
