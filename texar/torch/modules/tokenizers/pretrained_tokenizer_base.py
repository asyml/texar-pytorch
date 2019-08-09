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
Base class for pre-trained tokenizers.

The code structure adapted from:
    `https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_utils.py`
"""

from typing import Dict, List, Optional, Tuple

import os
import json
from abc import ABC
from pathlib import Path

from texar.torch.data.data_utils import maybe_download
from texar.torch.hyperparams import HParams
from texar.torch.module_base import ModuleBase
from texar.torch.modules.pretrained.pretrained_base import default_download_dir
from texar.torch.utils.types import MaybeList

__all__ = [
    "PretrainedTokenizerBase",
]

SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'


class PretrainedTokenizerBase(ModuleBase, ABC):
    r"""Base class inherited by all pre-trained tokenizer classes. This class
    handles downloading and loading pre-trained tokenizer and adding tokens to
    the vocabulary.

    Derived class can set up a few special tokens to be used in common scripts
    and internals: `bos_token`, `eos_token`, `unk_token`, `sep_token`,
    `pad_token`, `cls_token`, `mask_token`, and `additional_special_tokens`.

    We defined an `added_tokens_encoder` to add new tokens to the vocabulary
    without having to handle the specific vocabulary augmentation methods of
    the various underlying dictionary structures (BPE, sentencepiece...).
    """

    _MODEL_NAME = None
    _MODEL2URL = None
    _MAX_INPUT_SIZE = None
    _VOCAB_FILE_NAMES = None
    _SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token",
                                  "sep_token", "pad_token", "cls_token",
                                  "mask_token", "additional_special_tokens"]

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.sep_token = None
        self.pad_token = None
        self.cls_token = None
        self.mask_token = None
        self.additional_special_tokens = []

        self.max_len = int(1e12)
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        for key, value in self.hparams.items():
            if key in self._SPECIAL_TOKENS_ATTRIBUTES:
                setattr(self, key, value)

    def load_pretrained_tokenizer(self,
                                  pretrained_model_name: Optional[str] = None,
                                  cache_dir: Optional[str] = None,
                                  hparams=None):
        if not hasattr(self, "_hparams"):
            self._hparams = HParams(hparams, self.default_hparams())
        else:
            # Probably already parsed by subclasses. We rely on subclass
            # implementations to get this right.
            # As a sanity check, we require `hparams` to be `None` in this case.
            if hparams is not None:
                raise ValueError(
                    "`self._hparams` is already assigned, but `hparams` "
                    "argument is not None.")

        self.pretrained_model_dir = None

        if pretrained_model_name is None:
            pretrained_model_name = self._hparams.pretrained_model_name
        if pretrained_model_name is not None:
            self.pretrained_model_dir = self.download_checkpoint(
                pretrained_model_name, cache_dir)

    @classmethod
    def download_checkpoint(cls, pretrained_model_name: str,
                            cache_dir: Optional[str] = None) -> str:
        r"""Download the specified pre-trained checkpoint, and return the
        directory in which the checkpoint is cached.

        Args:
            pretrained_model_name (str): Name of the model checkpoint.
            cache_dir (str, optional): Path to the cache directory. If `None`,
                uses the default directory given by
                :meth:`~default_download_dir`.

        Returns:
            Path to the cache directory.
        """
        if pretrained_model_name in cls._MODEL2URL:
            download_path = cls._MODEL2URL[pretrained_model_name]
        else:
            raise ValueError(
                f"Pre-trained model not found: {pretrained_model_name}")

        if cache_dir is None:
            cache_path = default_download_dir(cls._MODEL_NAME)
        else:
            cache_path = Path(cache_dir)
        cache_path = cache_path / pretrained_model_name

        if not cache_path.exists():
            if isinstance(download_path, str):
                filename = download_path.split('/')[-1]
                maybe_download(download_path, cache_path, extract=True)
                folder = None
                for file in cache_path.iterdir():
                    if file.is_dir():
                        folder = file
                assert folder is not None
                (cache_path / filename).unlink()
                for file in folder.iterdir():
                    file.rename(file.parents[1] / file.name)
                folder.rmdir()
            else:
                for path in download_path:
                    maybe_download(path, cache_path)
            print(f"Pre-trained {cls._MODEL_NAME} checkpoint "
                  f"{pretrained_model_name} cached to {cache_path}")
        else:
            print(f"Using cached pre-trained {cls._MODEL_NAME} checkpoint "
                  f"from {cache_path}.")

        return str(cache_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str,
                        hparams=None):
        r"""Instantiate a pre-trained tokenizer from pre-trained vocabulary
        files.

        Args:
            pretrained_model_path: Path/Directory to the pre-trained model
                vocab files.

        """
        vocab_files = {}
        # Look for the tokenizer main vocabulary files
        for file_id, file_name in cls._VOCAB_FILE_NAMES.items():
            if os.path.isdir(pretrained_model_path):
                # If a directory is provided we look for the standard file name
                full_file_name = os.path.join(pretrained_model_path, file_name)
            else:
                # If a path to a file is provided we use it (will only work
                # for non-BPE tokenizer using a single vocabulary file)
                full_file_name = pretrained_model_path
            if not os.path.exists(full_file_name):
                print("Can't find file {}. We won't load it.".format(
                    full_file_name))
                full_file_name = None
            vocab_files[file_id] = full_file_name

        # Look for the additional tokens files
        all_vocab_files_names = {
            'added_tokens_file': ADDED_TOKENS_FILE,
            'special_tokens_map_file': SPECIAL_TOKENS_MAP_FILE}

        # If a path to a file was provided, get the parent directory
        saved_directory = pretrained_model_path
        if os.path.exists(saved_directory) and not os.path.isdir(
                saved_directory):
            saved_directory = os.path.dirname(saved_directory)

        for file_id, file_name in all_vocab_files_names.items():
            full_file_name = os.path.join(saved_directory, file_name)
            if not os.path.exists(full_file_name):
                print("Can't find file {}. We won't load it.".format(
                    full_file_name))
                full_file_name = None
            vocab_files[file_id] = full_file_name

        if all(full_file_name is None for full_file_name in
               vocab_files.values()):
            raise ValueError("Can't find tokenizer files in {}.".format(
                saved_directory))

        kwargs = {'pretrained_model_name': None}
        if hparams is not None:
            for key, value in hparams.items():
                kwargs[key] = value

        added_tokens_file = vocab_files.pop('added_tokens_file', None)
        special_tokens_map_file = vocab_files.pop(
            'special_tokens_map_file', None)

        for args_name, file_path in vocab_files.items():
            if args_name not in kwargs:
                kwargs[args_name] = file_path

        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding="utf-8") as f:
                special_tokens_map = json.load(f)
            for key, value in special_tokens_map.items():
                if key not in kwargs:
                    kwargs[key] = value

        tokenizer = cls(hparams=kwargs)

        # Add supplementary tokens.
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as f:
                added_tok_encoder = json.load(f)
            added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
            tokenizer.added_tokens_encoder.update(added_tok_encoder)
            tokenizer.added_tokens_decoder.update(added_tok_decoder)

        return tokenizer

    def save_pretrained(self, save_directory: str) -> Tuple[str, str, str]:
        r"""Save the tokenizer vocabulary files (with added tokens) and the
        `special-tokens-to-class-attributes-mapping` to a directory, so that it
        can be re-loaded using the :meth:`~from_pretrained`.
        """
        if not os.path.isdir(save_directory):
            raise ValueError("Saving directory ({}) should be a "
                             "directory".format(save_directory))

        special_tokens_map_file = os.path.join(save_directory,
                                               SPECIAL_TOKENS_MAP_FILE)
        added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)

        with open(special_tokens_map_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

        with open(added_tokens_file, 'w', encoding='utf-8') as f:
            if self.added_tokens_encoder:
                out_str = json.dumps(self.added_tokens_encoder,
                                     ensure_ascii=False)
            else:
                out_str = u"{}"
            f.write(out_str)

        vocab_files = self.save_vocabulary(save_directory)
        return vocab_files + (special_tokens_map_file, added_tokens_file)

    def save_vocabulary(self, save_directory: str) -> str:
        r"""Save the tokenizer vocabulary to a directory. This method doesn't
        save added tokens and special token mappings.

        Please use :meth:`~save_pretrained` to save the full tokenizer state so
        that it can be reloaded using :meth:`~from_pretrained`.
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.vocab_size + len(self.added_tokens_encoder)

    def add_tokens(self, new_tokens: List[Optional[str]]) -> int:
        r"""Add a list of new tokens to the tokenizer class. If the new tokens
        are not in the vocabulary, they are added to the `added_tokens_encoder`
        with indices starting from the last index of the current vocabulary.

        Args:
            new_tokens: A list of new tokens.

        Returns:
            Number of tokens added to the vocabulary which can be used to
            correspondingly increase the size of the associated model embedding
            matrices.
        """
        if not new_tokens:
            return 0

        to_add_tokens = []
        for token in new_tokens:
            if (self.convert_tokens_to_ids(token) ==
                    self.convert_tokens_to_ids(self.unk_token)):
                to_add_tokens.append(token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in
                                 enumerate(to_add_tokens))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        return len(to_add_tokens)

    def add_special_tokens(self,
                           special_tokens_dict: Dict[str, str]) -> \
            int:
        r"""Add a dictionary of special tokens (eos, pad, cls...) to the
        encoder and link them to class attributes. If the special tokens are
        not in the vocabulary, they are added to it and indexed starting from
        the last index of the current vocabulary.

        Args:
            special_tokens_dict: A dictionary of special tokens.

        Returns:
            Number of tokens added to the vocabulary which can be used to
            correspondingly increase the size of the associated model embedding
            matrices.
        """
        if not special_tokens_dict:
            return 0

        added_special_tokens = self.add_tokens(
            list(special_tokens_dict.values()))
        for key, value in special_tokens_dict.items():
            setattr(self, key, value)

        return added_special_tokens

    def tokenize(self, text: Optional[str], **kwargs) -> List[Optional[str]]:
        r"""Converts a string in a sequence of tokens (string), using the
        tokenizer. Split in words for word-based vocabulary or sub-words for
        sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Take care of added tokens.
        """
        def split_on_tokens(tok_list, string):
            if not string:
                return []
            if not tok_list:
                return self._tokenize(string, **kwargs)
            tok = tok_list[0]
            split_text = string.split(tok)
            return sum((split_on_tokens(tok_list[1:], sub_text.strip()) + [tok]
                        for sub_text in split_text), [])[:-1]

        added_tokens = list(
            self.added_tokens_encoder.keys()) + self.all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        r"""Converts a string in a sequence of tokens (string), using the
        tokenizer. Split in words for word-based vocabulary or sub-words for
        sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Don't take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: MaybeList[str]) -> MaybeList[int]:
        r"""Converts a single token or a sequence of tokens (str/unicode) in
        a integer id (resp.) a sequence of ids, using the vocabulary.
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified "
                "maximum sequence length for this model ({} > {}). Running "
                "this sequence through the model will result in indexing "
                "errors".format(len(ids), self.max_len))
        return ids

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token: str) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> MaybeList[int]:
        r"""Converts a string in a sequence of ids (integer), using the
        tokenizer and vocabulary. Same as
        `self.convert_tokens_to_ids(self.tokenize(text))`.
        """
        return self.convert_tokens_to_ids(self.tokenize(text))

    def convert_ids_to_tokens(self,
                              token_ids: MaybeList[int],
                              skip_special_tokens: bool = False) -> \
            MaybeList[int]:
        r"""Converts a single index or a sequence of indices (integers) in a
        token (resp.) a sequence of tokens (str/unicode), using the vocabulary
        and added tokens.

        Args:
            token_ids: A list of token ids.
            skip_special_tokens: Don't decode special tokens
                (`self.all_special_tokens`). Default: False
        """
        if isinstance(token_ids, int):
            if token_ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[token_ids]
            else:
                return self._convert_id_to_token(token_ids)
        tokens = []
        for index in token_ids:
            if index in self.all_special_ids and skip_special_tokens:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, token_id: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        r"""Converts a sequence of tokens (string) in a single string.
        The most simple way to do it is `' '.join(tokens)`, but we often want
        to remove sub-word tokenization artifacts at the same time.
        """
        return ' '.join(tokens)

    def decode(self, token_ids: List[int],
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = True) -> str:
        r"""Converts a sequence of ids (integer) in a string, using the
        tokenizer and vocabulary with options to remove special tokens and
        clean up tokenization spaces.
        """
        filtered_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens)
        text = self.convert_tokens_to_string(filtered_tokens)
        if clean_up_tokenization_spaces:
            text = clean_up_tokenization(text)
        return text

    @classmethod
    def available_checkpoints(cls) -> List[str]:
        return list(cls._MODEL2URL.keys())

    @property
    def special_tokens_map(self) -> Dict[str, str]:
        r"""A dictionary mapping special token class attribute
        (`cls_token`, `unk_token` ...) to their values (`<unk>`, `<cls>`, ...)
        """
        set_attr = {}
        for attr in self._SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        r"""List all the special tokens (`<unk>`, `<cls>`, ...) mapped to class
        attributes (`cls_token`, `unk_token`, ...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (
                attr_value if isinstance(attr_value, (list, tuple)) else [
                    attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        r"""List the vocabulary indices of the special tokens
        (`<unk>`, `<cls>`, ...) mapped to class attributes
        (`cls_token`, `unk_token` ...).
        """
        all_toks = self.all_special_tokens
        all_ids = list(self.convert_tokens_to_ids(t) for t in all_toks)
        return all_ids


def clean_up_tokenization(out_string: str) -> str:
    out_string = out_string.replace(' .', '.').replace(' ?', '?').\
        replace(' !', '!').replace(' ,', ',').replace(" ' ", "'").\
        replace(" n't", "n't").replace(" 'm", "'m").\
        replace(" do not", " don't").replace(" 's", "'s").\
        replace(" 've", "'ve").replace(" 're", "'re")
    return out_string
