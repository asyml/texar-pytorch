import math
from typing import Sequence

from texar.torch.run.metric.base_metric import StreamingMetric

__all__ = [
    "PearsonR",
]


class PearsonR(StreamingMetric[float, float]):
    x_sum: float
    x2_sum: float
    y_sum: float
    y2_sum: float
    xy_sum: float

    def reset(self) -> None:
        super().reset()
        self.x_sum = self.y_sum = 0.0
        self.x2_sum = self.y2_sum = 0.0
        self.xy_sum = 0.0

    def add(self, xs: Sequence[float], ys: Sequence[float]):
        super().add(xs, ys)
        self.x_sum += sum(xs)
        self.x2_sum += sum(x * x for x in xs)
        self.y_sum += sum(ys)
        self.y2_sum += sum(y * y for y in ys)
        self.xy_sum += sum(x * y for x, y in zip(xs, ys))

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        numerator = self.xy_sum - self.x_sum * self.y_sum / self.count
        denominator_x = self.x2_sum - self.x_sum ** 2 / self.count
        denominator_y = self.y2_sum - self.y_sum ** 2 / self.count
        if denominator_x == 0.0 or denominator_y == 0.0:
            return math.nan
        return numerator / math.sqrt(denominator_x * denominator_y)
