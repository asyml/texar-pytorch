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

import torch

import texar.torch as tx


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = tx.modules.XLNetDecoder(pretrained_model_name='xlnet-large-cased')
    model = model.to(device)

    tokenizer = tx.data.XLNetTokenizer(
        pretrained_model_name='xlnet-large-cased')

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
        r"""Splits list `xs` by value `y`.

        Example:
            list(split_by([1,2,4,5,6,4,7,4], 4))
            # [[1, 2], [5, 6], [7]]
        """
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
        print("=== Prompt ===")
        print(text)
        model.eval()
        text = text.replace("\n", "<eop>")
        tokens = pad_ids + tokenizer.map_text_to_id(text)
        tokens = torch.tensor(tokens, device=device).expand(n_samples, -1)
        kwargs.setdefault("print_steps", True)
        decode_output, _ = model(start_tokens=tokens,
                                 end_token=eod_id,
                                 max_decoding_length=length,
                                 **kwargs)
        decode_samples = decode_output.sample_id.tolist()
        for idx, sample_tokens in enumerate(decode_samples):
            print(f"=== Sample {idx} ===")
            output = "\n".join(tokenizer.map_id_to_text(xs) for xs in split_by(
                sample_tokens, tokenizer.map_token_to_id("<eop>")))
            print(output)

    try:
        from IPython import embed
        print("Generate text by calling: sample(\"<your prompt text>\", ...).\n"
              "For options, refer to `forward` method of `XLNetDecoder`.\n")
        embed()
    except ImportError:
        print("To be able to specify sampling options, please install IPython.")
        try:
            while True:
                prompt = input("Input prompt: ")
                print(sample(prompt))
        except EOFError:
            pass


if __name__ == '__main__':
    main()
