from typing import Any, Optional

from .. import Tensor


class Parameter(Tensor):
    def __new__(cls, data: Optional[Any] = None, requires_grad:bool = True): ...
