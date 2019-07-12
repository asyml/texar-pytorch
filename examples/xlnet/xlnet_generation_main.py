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
                    help="The number of samples per input.")
parser.add_argument('--batch_size', type=int, default=1,
                    help="The batch size of input.")
parser.add_argument('--max_decoding_length', type=int, default=128,
                    help="The maximun length of generated text.")
parser.add_argument('--temperature', type=float, default=0.7,
                    help="Softmax temperature for top-k sample decoding. Must "
                         "be strictly greater than 0. Defaults to 0.7.")
parser.add_argument('--top_k', type=int, default=40,
                    help="The number of top most likely candidates from a vocab "
                         "distribution.")
parser.add_argument('--is_interactive', action='store_true',
                    help="Interactive mode or not.")
parser.add_argument('--sentence_piece', type=str,
                    default="pretrained/xlnet_cased_L-24_H-1024_A-16/"
                            "spiece.model",
                    help="Location to load sentene piece model from")

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
        In 1991, the remains of Russian Tsar Nicholas II and his family
        (except for Alexei and Maria) are discovered.
        The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich,
        narrates the remainder of the story. 1883 Western Siberia, a young
        Grigori Rasputin is asked by his father and a group of men to perform
        magic. Rasputin has a vision and denounces one of the men as a horse
        thief. Although his father initially slaps him for making such an
        accusation, Rasputin watches as the man is chased outside and beaten.
        Twenty years later, Rasputin sees a vision of the Virgin Mary,
        prompting him to become a priest. Rasputin quickly becomes famous,
        with people, even a bishop, begging for his blessing. """
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

    def sample(text: str, length: int = 200, n_samples=3, **kwargs):
        print("=== Prompt ===")
        print(text)
        model.eval()
        text = text.replace("\n", "<eop>")
        tokens = pad_ids + tokenize_fn(text)
        tokens = torch.tensor(tokens, device=device).expand(n_samples, -1)
        kwargs.setdefault("print_steps", True)
        decode_output, _ = model(
            tokens, max_decoding_length=length, **kwargs)
        decode_samples = decode_output.sample_id.tolist()
        for idx, sample_tokens in enumerate(decode_samples):
            print(f"=== Sample {idx} ===")
            output = "\n".join(sp_model.DecodeIds(xs) for xs in split_by(
                sample_tokens, xlnet.data.utils.special_symbols["<eop>"]))
            print(output)

    if args.is_interactive:
        try:
            from IPython import embed
            print("Generate text by calling: "
                  "sample(\"<your prompt text>\", ...).\nFor options, refer to "
                  "`decode` method of `XLNetDecoder`.\n")
            embed()
        except ImportError:
            print("To be able to specify sampling options, please install "
                  "IPython.")
            try:
                while True:
                    prompt = input("Input prompt: ")
                    print(sample(prompt))
            except EOFError:
                pass
    else:
        # Generate samples from scratch
        for _ in range(args.batch_size):
            sample(text="<BOS>", length=args.max_decoding_length,
                   n_samples=args.nsamples)


if __name__ == '__main__':
    main()
