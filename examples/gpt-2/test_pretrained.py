"""
Unit tests for pretrained models.
"""
import unittest
import torch
import texar.torch as tx


class PretrainedModel(unittest.TestCase):

    def test_equal(self):
        # default input for testing
        input_sentence = 'This is GPT-2 small. ' + \
        'It has 130M parameters and it is from OpenAI.'

        # texar output
        model = tx.modules.GPT2Decoder("gpt2-small")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        tokenizer = tx.data.GPT2Tokenizer(
            pretrained_model_name="gpt2-small")
        end_token = tokenizer.map_token_to_id('<|endoftext|>')

        context_tokens = tokenizer.map_text_to_id(input_sentence)
        context = torch.tensor(
            [context_tokens for _ in range(1)],
            device=device)
        context_length = torch.tensor(
            [len(context_tokens) for _ in range(1)],
            device=device)
        start_tokens = context[:, 0]

        def _get_helper(start_tokens):
            return tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=end_token,
                top_k=40,
                softmax_temperature=0.7)

        helper = _get_helper(start_tokens)
        output, _ = model(
            context=context,
            context_sequence_length=context_length,
            max_decoding_length=20,
            helper=helper)
        texar_logits = output.logits

        baseline = torch.tensor([[-39.3447, -40.9561, -42.0426, -40.7665],
        [-86.7560, -82.9948, -87.1625, -87.2382],
        [-97.7438, -102.7465, -104.9421, -104.8722],
        [-106.1100, -107.3229, -107.4181, -109.5231],
        [-103.8968, -105.6753, -104.8125, -108.4072]])

        self.assertTrue(torch.allclose(
            texar_logits[0, 0:20:4, 10000:50000:10000], baseline))


if __name__ == "__main__":
    unittest.main()
