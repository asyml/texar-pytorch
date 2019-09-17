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

import texar.torch as tx

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None,
                    help="Checkpoint to load model weights from.")
parser.add_argument("--pretrained-model-name", type=str,
                    default="xlnet-large-cased",
                    help="The pre-trained model to load selected in the list "
                         "of: `xlnet-base-cased`, `xlnet-large-cased`.")
parser.add_argument('--seed', type=int, default=None, help="Random seed.")
parser.add_argument('--nsamples', type=int, default=1,
                    help="Total number of samples to generate. Used in "
                         "non-interactive mode.")
parser.add_argument('--batch-size', type=int, default=1,
                    help="The batch size of input.")
parser.add_argument('--max-decoding-length', type=int, default=100,
                    help="The maximun length of generated text.")
parser.add_argument('--temperature', type=float, default=0.7,
                    help="Softmax temperature for top-k sample decoding. Must "
                         "be strictly greater than 0. Defaults to 0.7.")
parser.add_argument('--top-k', type=int, default=40,
                    help="The number of top most likely candidates to choose "
                         "from at each step. This is use "
                         "TopKSampleEmbeddingHelper for decoding. Ignored if "
                         "'p' is given.")
parser.add_argument('--top-p', type=float, default=None,
                    help="Select tokens with cumulative probability of at most "
                         "'top-p' when arranged in decreasing order. This "
                         "will use TopPSampleEmbeddingHelper for decoding.")
parser.add_argument('--interactive', action='store_true',
                    help="Interactive mode or not.")

args = parser.parse_args()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = tx.modules.XLNetDecoder(
        pretrained_model_name=args.pretrained_model_name)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    model = model.to(device)

    tokenizer = tx.data.XLNetTokenizer(
        pretrained_model_name=args.pretrained_model_name)

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
    pad_ids = tokenizer.map_text_to_id(pad_txt)
    eod_id = tokenizer.map_token_to_id("<eod>")
    pad_ids.append(eod_id)

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
        tokens = pad_ids + tokenizer.map_text_to_id(text)
        tokens = torch.tensor(tokens, device=device).expand(n_samples, -1)
        if args.top_p:
            kwargs["p"] = args.top_p
            decode_output, _ = model(
                start_tokens=tokens,
                end_token=eod_id,
                max_decoding_length=length,
                print_steps=True,
                helper_type=tx.modules.TopPSampleEmbeddingHelper,
                **kwargs)
        else:
            kwargs["top_k"] = args.top_k
            decode_output, _ = model(
                start_tokens=tokens,
                end_token=eod_id,
                max_decoding_length=length,
                print_steps=True,
                helper_type=tx.modules.TopKSampleEmbeddingHelper,
                **kwargs)
        decode_samples = decode_output.sample_id.tolist()
        for idx, sample_tokens in enumerate(decode_samples):
            print(f"=== Sample {idx} ===")
            output = "\n".join(tokenizer.map_id_to_text(xs) for xs in split_by(
                sample_tokens, tokenizer.map_token_to_id("<eop>")))
            print(output)

    nsamples = args.nsamples
    batch_size = args.batch_size
    max_decoding_length = args.max_decoding_length
    assert nsamples % batch_size == 0, (
        "nsamples must be dividable by batch_size")

    if args.interactive:
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
