import os
import unittest

pretrained_test = unittest.skipUnless(
    os.environ.get('TEST_PRETRAINED', 0) or os.environ.get('TEST_ALL', 0),
    "Test requires loading pre-trained checkpoints. "
    "Set `TEST_PRETRAINED=1` or `TEST_ALL=1` to run.")
data_test = unittest.skipUnless(
    os.environ.get('TEST_DATA', 0) or os.environ.get('TEST_ALL', 0),
    "Test requires loading large data files. "
    "Set `TEST_DATA=1` or `TEST_ALL=1` to run.")
