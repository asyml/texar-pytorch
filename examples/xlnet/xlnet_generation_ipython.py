import sentencepiece as spm
import torch

import xlnet
import xlnet.model.decoder


def main():
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = 'cpu'

    model = xlnet.model.decoder.XLNetDecoder()
    xlnet.model.load_from_tf_checkpoint(
        model, "pretrained/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt")
    model = model.to(device)
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load("pretrained/xlnet_cased_L-24_H-1024_A-16/spiece.model")
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
        decode_output, _ = model.decode(
            tokens, max_decoding_length=length, **kwargs)
        decode_samples = decode_output.sample_id.tolist()
        for idx, sample_tokens in enumerate(decode_samples):
            print(f"=== Sample {idx} ===")
            output = "\n".join(sp_model.DecodeIds(xs) for xs in split_by(
                sample_tokens, xlnet.data.utils.special_symbols["<eop>"]))
            print(output)

    try:
        from IPython import embed
        print("Generate text by calling: sample(\"<your prompt text>\", ...).\n"
              "For options, refer to `decode` method of `XLNetDecoder`.\n")
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
