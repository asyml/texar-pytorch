from typing import List

from texar.torch.run.metric.base_metric import StreamingMetric


class METEOR(StreamingMetric[str, float]):

    def reset(self) -> None:
        pass

    def add(self, predicted: List[str], labels: List[str]) -> None:
        pass

    def value(self) -> float:
        pass


class BLEU(StreamingMetric[str, float]):
    def reset(self) -> None:
        pass

    def add(self, predicted: List[str], labels: List[str]) -> None:
        pass

    def value(self) -> float:
        pass
