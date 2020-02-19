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
"""
Unit tests for ELMo Encoder.

Code adapted from:
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/modules/elmo_test.py`
"""

import unittest

from texar.torch.data.tokenizers.elmo_tokenizer_utils import batch_to_ids
from texar.torch.modules.encoders.elmo_encoder import ELMoEncoder
from texar.torch.utils.test import pretrained_test


class ELMoEncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.ELMoEncoder` class.
    """

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        sentences = [
            ["The", "sentence", "."],
            ["ELMo", "helps", "disambiguate", "ELMo", "from", "Elmo", "."],
        ]
        character_ids = batch_to_ids(sentences)
        for pretrained_model_name in ELMoEncoder.available_checkpoints():
            encoder = ELMoEncoder(pretrained_model_name=pretrained_model_name)
            _ = encoder(character_ids)

    def test_encode(self):
        r"""Tests encoding.
        """
        hparams = {
            "pretrained_model_name": None,
            'encoder': {
                "lstm": {
                    "cell_clip": 3,
                    "use_skip_connections": True,
                    "n_layers": 2,
                    "proj_clip": 3,
                    "projection_dim": 16,
                    "dim": 64
                },
                "char_cnn": {
                    "embedding": {
                        "dim": 4
                    },
                    "filters": [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                    "n_highway": 2,
                    "n_characters": 262,
                    "max_characters_per_token": 50,
                    "activation": "relu"
                }
            }
        }
        encoder = ELMoEncoder(hparams=hparams)

        sentences = [
            ["The", "sentence", "."],
            ["ELMo", "helps", "disambiguate", "ELMo", "from", "Elmo", "."],
        ]
        character_ids = batch_to_ids(sentences)
        output = encoder(character_ids)
        elmo_representations = output["elmo_representations"]
        mask = output["mask"]

        assert len(elmo_representations) == 2
        assert list(elmo_representations[0].size()) == [2, 7, 32]
        assert list(elmo_representations[1].size()) == [2, 7, 32]
        assert list(mask.size()) == [2, 7]

    def test_elmo_keep_sentence_boundaries(self):
        hparams = {
            "pretrained_model_name": None,
            'encoder': {
                "lstm": {
                    "cell_clip": 3,
                    "use_skip_connections": True,
                    "n_layers": 2,
                    "proj_clip": 3,
                    "projection_dim": 16,
                    "dim": 64
                },
                "char_cnn": {
                    "embedding": {
                        "dim": 4
                    },
                    "filters": [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                    "n_highway": 2,
                    "n_characters": 262,
                    "max_characters_per_token": 50,
                    "activation": "relu"
                }
            },
            'dropout': 0.0,
            'keep_sentence_boundaries': True,
        }
        encoder = ELMoEncoder(hparams=hparams)

        sentences = [
            ["The", "sentence", "."],
            ["ELMo", "helps", "disambiguate", "ELMo", "from", "Elmo", "."],
        ]
        character_ids = batch_to_ids(sentences)
        output = encoder(character_ids)
        elmo_representations = output["elmo_representations"]
        mask = output["mask"]

        assert len(elmo_representations) == 2
        # Add 2 to the lengths because we're keeping the start and end of
        # sentence tokens.
        assert list(elmo_representations[0].size()) == [2, 7 + 2, 32]
        assert list(elmo_representations[1].size()) == [2, 7 + 2, 32]
        assert list(mask.size()) == [2, 7 + 2]

    @pretrained_test
    def test_trainable_variables(self):
        encoder = ELMoEncoder()
        elmo_grads = [
            param.requires_grad for param in encoder._elmo_lstm.parameters()
        ]
        assert all(grad is False for grad in elmo_grads)

        encoder = ELMoEncoder(hparams={'requires_grad': True})
        elmo_grads = [
            param.requires_grad for param in encoder._elmo_lstm.parameters()
        ]
        assert all(grad is True for grad in elmo_grads)


if __name__ == "__main__":
    unittest.main()
