import sentencepiece as spm

import xlnet


def main():
    model = xlnet.model.model.XLNetLM()
    xlnet.model.load_from_tf_checkpoint(
        model, "pretrained/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt")
    model = model.cuda()
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

    def sample(text: str, length: int = 200, **kwargs):
        model.eval()
        text = text.replace("\n", "<eop>")
        tokens = tokenize_fn(text)
        decode_tokens, _ = model.decode(
            pad_ids + tokens, max_length=(len(pad_ids + tokens) + length),
            **kwargs)
        decode_tokens = decode_tokens[len(pad_ids):]
        output = "\n\n".join(sp_model.DecodeIds(xs) for xs in split_by(
            decode_tokens, xlnet.data.utils.special_symbols["<eop>"]))
        print(output)

    try:
        from IPython import embed
        print("Generate text by calling: sample(\"<your prompt text>\", ...).\n"
              "For options, refer to `decode` method of `XLNetLM`.\n")
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
