import os
import unittest

__all__ = [
    "pretrained_test",
    "data_test",
    "external_library_test",
]


def define_skip_condition(flag: str, explanation: str):
    return unittest.skipUnless(
        os.environ.get(flag, 0) or os.environ.get('TEST_ALL', 0),
        explanation + f" Set `{flag}=1` or `TEST_ALL=1` to run.")


pretrained_test = define_skip_condition(
    'TEST_PRETRAINED', "Test requires loading pre-trained checkpoints.")
data_test = define_skip_condition(
    'TEST_DATA', "Test requires loading large data files.")


def external_library_test(name: str):
    import importlib
    try:
        importlib.import_module(name)
        return lambda x: x  # no changes
    except ImportError:
        return unittest.skip(
            f"Test requires external library {name} to be installed.")
