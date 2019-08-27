import unittest

from texar.torch.run.metric.generation import *
from texar.torch.evals.bleu import corpus_bleu


class GenerationMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hypotheses = [
            "this is a test sentence to evaluate the good bleu score . 词",
            "i believe that that the script is 词 perfectly correct .",
            "this score should be pretty bad".split(),
        ]
        self.references = [
            "this is a test sentence to evaluate the good score .",
            "i believe that the script is perfectly correct .".split(),
            "yeah , this is a totally different sentence .",
        ]

    def test_bleu(self):
        metric = BLEU(pred_name="", label_name="")
        for idx, (hyp, ref) in enumerate(zip(self.hypotheses, self.references)):
            metric.add([hyp], [ref])
            value = metric.value()
            answer = corpus_bleu([[r] for r in self.references[:(idx + 1)]],
                                 self.hypotheses[:(idx + 1)])
            self.assertAlmostEqual(value, answer)
