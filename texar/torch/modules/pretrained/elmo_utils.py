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
Utils of ELMo Modules.

Code adapted from:
    `https://github.com/allenai/allennlp/blob/master/allennlp/common/checks.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo_lstm.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/modules/encoder_base.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/modules/lstm_cell_with_projection.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/modules/time_distributed.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/embedding.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/nn/initializers.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py`
"""
import itertools
import json
import logging

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy
import torch

from torch.nn import ParameterList, Parameter
from torch.nn.functional import embedding
from torch.nn.utils.rnn import (
    pad_packed_sequence, pack_padded_sequence, PackedSequence)

from texar.torch.data.tokenizers.elmo_tokenizer_utils import (
    batch_to_ids, ELMoCharacterMapper)
from texar.torch.utils.utils import (
    lazy_groups_of, sort_batch_by_length)

# pylint: disable=attribute-defined-outside-init,protected-access

__all__ = [
    "_ElmoBiLm",
    "_ElmoCharacterEncoder",
    "_EncoderBase",
    "ConfigurationError",
    "ElmoLstm",
    "Embedding",
    "Highway",
    "LstmCellWithProjection",
    "ScalarMix",
    "TimeDistributed",
    "add_sentence_boundary_token_ids",
    "block_orthogonal",
    "combine_initial_dims",
    "get_device_of",
    "get_dropout_mask",
    "remove_sentence_boundaries",
    "uncombine_initial_dims",
]


class _ElmoBiLm(torch.nn.Module):
    r"""Run a pre-trained bidirectional language model, outputting the
    activations at each layer for weighting together into an ELMo
    representation (with `allennlp.modules.seq2seq_encoders.Elmo`).
    This is a lower level class, useful for advanced uses, but most users
    should use `allennlp.modules.Elmo` directly.

    # Parameters

    options_file : `str`
        ELMo JSON options file
    weight_file : `str`
        ELMo hdf5 weight file
    requires_grad : `bool`, optional, (default = False).
        If True, compute gradient of ELMo parameters for fine tuning.
    vocab_to_cache : `List[str]`, optional, (default = None).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, _ElmoBiLm expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    """

    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = False,
        vocab_to_cache: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self._token_embedder = _ElmoCharacterEncoder(
            options_file, weight_file, requires_grad=requires_grad
        )

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning(
                "You are fine tuning ELMo and caching char CNN word vectors. "
                "This behaviour is not guaranteed to be well defined, "
                "particularly. "
                "if not all of your inputs will occur in the vocabulary cache."
            )
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None  # type: ignore
        self._eos_embedding: torch.Tensor = None  # type: ignore
        if vocab_to_cache:
            logging.info(
                "Caching character cnn layers for words in vocabulary.")
            # This sets 3 attributes, _word_embedding, _bos_embedding and
            # _eos_embedding. They are set in the method so they can be accessed
            # from outside the constructor.
            self.create_cached_cnn_embeddings(vocab_to_cache)

        with open(options_file, "r") as fin:
            options = json.load(fin)
        if not options["lstm"].get("use_skip_connections"):
            raise ConfigurationError(
                "We only support pretrained biLMs with residual connections")
        self._elmo_lstm = ElmoLstm(
            input_size=options["lstm"]["projection_dim"],
            hidden_size=options["lstm"]["projection_dim"],
            cell_size=options["lstm"]["dim"],
            num_layers=options["lstm"]["n_layers"],
            memory_cell_clip_value=options["lstm"]["cell_clip"],
            state_projection_clip_value=options["lstm"]["proj_clip"],
            requires_grad=requires_grad,
        )

        if weight_file is not None:
            self._elmo_lstm.load_weights(weight_file)
        # Number of representation layers including context independent layer
        self.num_layers = options["lstm"]["n_layers"] + 1

    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()

    def forward(  # type: ignore
        self, inputs: torch.Tensor, word_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        r"""Encodes the inputs.

        # Parameters

        inputs : `torch.Tensor`, required.
            Shape `(batch_size, timesteps, 50)` of character ids representing
            the current batch.
        word_inputs : `torch.Tensor`, required.
            If you passed a cached vocab, you can in addition pass a tensor of
            shape `(batch_size, timesteps)`, which represent word ids which
            have been pre-cached.

        # Returns

        Dict with keys:

        `'activations'` : `List[torch.Tensor]`
            A list of activations at each layer of the network, each of shape
            `(batch_size, timesteps + 2, embedding_dim)`
        `'mask'`:  `torch.Tensor`
            Shape `(batch_size, timesteps + 2)` long tensor with sequence mask.

        Note that the output tensors all include additional special begin and
        end of sequence markers.
        """
        if self._word_embedding is not None and word_inputs is not None:
            try:
                mask_without_bos_eos = (word_inputs > 0).long()
                # The character cnn part is cached - just look it up.
                embedded_inputs = self._word_embedding(
                    word_inputs)
                # shape (batch_size, timesteps + 2, embedding_dim)
                type_representation, mask = add_sentence_boundary_token_ids(
                    embedded_inputs, mask_without_bos_eos, self._bos_embedding,
                    self._eos_embedding
                )
            except RuntimeError:
                # Back off to running the character convolutions,
                # as we might not have the words in the cache.
                token_embedding = self._token_embedder(inputs)
                mask = token_embedding["mask"]
                type_representation = token_embedding["token_embedding"]
        else:
            token_embedding = self._token_embedder(inputs)
            mask = token_embedding["mask"]
            type_representation = token_embedding["token_embedding"]
        lstm_outputs = self._elmo_lstm(type_representation, mask)

        # Prepare the output.  The first layer is duplicated.
        # Because of minor differences in how masking is applied depending
        # on whether the char cnn layers are cached, we'll be defensive and
        # multiply by the mask here. It's not strictly necessary, as the
        # mask passed on is correct, but the values in the padded areas
        # of the char cnn representations can change.
        output_tensors = [
            torch.cat([type_representation, type_representation], dim=-1)
            * mask.float().unsqueeze(-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs,
                                             lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {"activations": output_tensors, "mask": mask}

    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None:
        r"""Given a list of tokens, this method precomputes word representations
        by running just the character convolutions and highway layers of elmo,
        essentially creating uncontextual word vectors. On subsequent forward
        passes, the word ids are looked up from an embedding, rather than being
        computed on the fly via the CNN encoder.

        This function sets 3 attributes:

        _word_embedding : `torch.Tensor`
            The word embedding for each word in the tokens passed to this
            method.
        _bos_embedding : `torch.Tensor`
            The embedding for the BOS token.
        _eos_embedding : `torch.Tensor`
            The embedding for the EOS token.

        # Parameters

        tokens : `List[str]`, required.
            A list of tokens to precompute character convolutions for.
        """
        tokens = [ELMoCharacterMapper.bos_token,
                  ELMoCharacterMapper.eos_token] + tokens
        timesteps = 32
        batch_size = 32
        chunked_tokens = lazy_groups_of(iter(tokens), timesteps)

        all_embeddings = []
        device = get_device_of(next(self.parameters()))
        for batch in lazy_groups_of(chunked_tokens, batch_size):
            # Shape (batch_size, timesteps, 50)
            batched_tensor = batch_to_ids(batch)
            # NOTE: This device check is for when a user calls this method
            # having already placed the model on a device. If this is called in
            # the constructor, it will probably happen on the CPU. This isn't
            # too bad, because it's only a few convolutions and will likely
            # be very fast.
            if device >= 0:
                batched_tensor = batched_tensor.cuda(device)
            output = self._token_embedder(batched_tensor)
            token_embedding = output["token_embedding"]
            mask = output["mask"]
            token_embedding, _ = remove_sentence_boundaries(token_embedding,
                                                            mask)
            all_embeddings.append(token_embedding.view(
                -1, token_embedding.size(-1)))
        full_embedding = torch.cat(all_embeddings, 0)

        # We might have some trailing embeddings from padding in the batch, so
        # we clip the embedding and lookup to the right size.
        full_embedding = full_embedding[: len(tokens), :]
        embedding_ = full_embedding[2: len(tokens), :]
        vocab_size, embedding_dim = list(embedding_.size())

        self._bos_embedding = full_embedding[0, :]
        self._eos_embedding = full_embedding[1, :]
        self._word_embedding = Embedding(  # type: ignore
            vocab_size,
            embedding_dim,
            weight=embedding_.data,
            trainable=self._requires_grad,
            padding_index=0,
        )


class _ElmoCharacterEncoder(torch.nn.Module):
    r"""Compute context insensitive token representation using pretrained biLM.

    This embedder has input character ids of size
    (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where
    embedding_dim is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence
    corresponding to <S> and </S>, the beginning and end of sentence tokens.

    Note: this is a lower level class useful for advanced usage.  Most users
    should use `ElmoTokenEmbedder` or `allennlp.modules.Elmo` instead.

    # Parameters

    options_file : `str`
        ELMo JSON options file
    weight_file : `str`
        ELMo hdf5 weight file
    requires_grad : `bool`, optional, (default = False).
        If True, compute gradient of ELMo parameters for fine tuning.

    The relevant section of the options file is something like:
    .. example-code::

        .. code-block:: python

            {'char_cnn': {
                'activation': 'relu',
                'embedding': {'dim': 4},
                'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                'max_characters_per_token': 50,
                'n_characters': 262,
                'n_highway': 2
                }
            }
    """

    def __init__(self, options_file: str, weight_file: str,
                 requires_grad: bool = False) -> None:
        super().__init__()

        with open(options_file, "r") as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file

        self.output_dim = self._options["lstm"]["projection_dim"]
        self.requires_grad = requires_grad

        if weight_file is not None:
            self._load_weights()
        else:
            # Do not load the weights
            self._load_weights(False)

        # Cache the arrays for use in forward -- +1 due to masking.
        self._beginning_of_sentence_characters = torch.from_numpy(
            numpy.array(
                ELMoCharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = torch.from_numpy(
            numpy.array(ELMoCharacterMapper.end_of_sentence_characters) + 1
        )

    def get_output_dim(self):
        return self.output_dim

    def forward(self,  # type: ignore
                inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Compute context insensitive token embeddings for ELMo
        representations.

        # Parameters

        inputs : `torch.Tensor`
            Shape `(batch_size, sequence_length, 50)` of character ids
            representing the current batch.

        # Returns

        Dict with keys:
        `'token_embedding'` : `torch.Tensor`
            Shape `(batch_size, sequence_length + 2, embedding_dim)` tensor
            with context insensitive token representations.
        `'mask'`:  `torch.Tensor`
            Shape `(batch_size, sequence_length + 2)` long tensor with
            sequence mask.
        """
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        character_ids_with_bos_eos, mask_with_bos_eos = \
            add_sentence_boundary_token_ids(
                inputs, mask, self._beginning_of_sentence_characters,
                self._end_of_sentence_characters)

        # the character id embedding
        max_chars_per_token = \
            self._options["char_cnn"]["max_characters_per_token"]
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
            character_ids_with_bos_eos.view(-1, max_chars_per_token),
            self._char_embedding_weights)

        # run convolutions
        cnn_options = self._options["char_cnn"]
        activation: Callable
        if cnn_options["activation"] == "tanh":
            activation = torch.tanh
        elif cnn_options["activation"] == "relu":
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, "char_conv_{}".format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return {
            "mask": mask_with_bos_eos,
            "token_embedding": token_embedding.view(batch_size,
                                                    sequence_length, -1),
        }

    def _load_weights(self, load_weights=True):
        self._load_char_embedding(load_weights)
        self._load_cnn_weights(load_weights)
        self._load_highway(load_weights)
        self._load_projection(load_weights)

    def _load_char_embedding(self, load_weights):

        if load_weights:
            with h5py.File(self._weight_file, "r") as fin:
                char_embed_weights = fin["char_embed"][...]

            weights = numpy.zeros(
                (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
                dtype="float32"
            )
            weights[1:, :] = char_embed_weights

            self._char_embedding_weights = torch.nn.Parameter(
                torch.FloatTensor(weights), requires_grad=self.requires_grad
            )
        else:
            weights = numpy.zeros(
                (self._options['char_cnn']['n_characters'],
                 self._options['char_cnn']['embedding']['dim']),
                dtype="float32"
            )
            self._char_embedding_weights = torch.nn.Parameter(
                torch.FloatTensor(weights), requires_grad=self.requires_grad
            )

    def _load_cnn_weights(self, load_weights):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        char_embed_dim = cnn_options["embedding"]["dim"]

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim, out_channels=num,
                kernel_size=width, bias=True
            )

            if load_weights:
                # load the weights
                with h5py.File(self._weight_file, "r") as fin:
                    weight = fin["CNN"]["W_cnn_{}".format(i)][...]
                    bias = fin["CNN"]["b_cnn_{}".format(i)][...]

                w_reshaped = numpy.transpose(weight.squeeze(axis=0),
                                             axes=(2, 1, 0))
                if w_reshaped.shape != tuple(conv.weight.data.shape):
                    raise ValueError("Invalid weight file")
                conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
                conv.bias.data.copy_(torch.FloatTensor(bias))

                conv.weight.requires_grad = self.requires_grad
                conv.bias.requires_grad = self.requires_grad

            convolutions.append(conv)
            self.add_module("char_conv_{}".format(i), conv)

        self._convolutions = convolutions

    def _load_highway(self, load_weights):

        # the highway layers have same dimensionality as the number of cnn
        # filters
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options["n_highway"]

        # create the layers, and load the weights
        self._highways = Highway(n_filters, n_highway,
                                 activation=torch.nn.functional.relu)

        if load_weights:
            for k in range(n_highway):
                # The AllenNLP highway is one matrix multplication with
                # concatenation of transform and carry weights.
                with h5py.File(self._weight_file, "r") as fin:
                    # The weights are transposed due to multiplication order
                    # assumptions in tf vs pytorch (tf.matmul(X, W) vs
                    # pytorch.matmul(W, X))
                    w_transform = numpy.transpose(
                        fin["CNN_high_{}".format(k)]["W_transform"][...])
                    # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but
                    # tf is (1 - g) * x + g * f(x)
                    w_carry = -1.0 * numpy.transpose(
                        fin["CNN_high_{}".format(k)]["W_carry"][...])
                    weight = numpy.concatenate([w_transform, w_carry], axis=0)
                    self._highways._layers[k].weight.data.copy_(
                        torch.FloatTensor(weight))
                    self._highways._layers[k].weight.requires_grad = \
                        self.requires_grad

                    b_transform = \
                        fin["CNN_high_{}".format(k)]["b_transform"][...]
                    b_carry = \
                        -1.0 * fin["CNN_high_{}".format(k)]["b_carry"][...]
                    bias = numpy.concatenate([b_transform, b_carry], axis=0)
                    self._highways._layers[k].bias.data.copy_(
                        torch.FloatTensor(bias))
                    self._highways._layers[k].bias.requires_grad = \
                        self.requires_grad

    def _load_projection(self, load_weights):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)

        self._projection = torch.nn.Linear(n_filters, self.output_dim,
                                           bias=True)

        if load_weights:
            with h5py.File(self._weight_file, "r") as fin:
                weight = fin["CNN_proj"]["W_proj"][...]
                bias = fin["CNN_proj"]["b_proj"][...]
                self._projection.weight.data.copy_(torch.FloatTensor(
                    numpy.transpose(weight)))
                self._projection.bias.data.copy_(torch.FloatTensor(bias))
                self._projection.weight.requires_grad = self.requires_grad
                self._projection.bias.requires_grad = self.requires_grad


RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
RnnStateStorage = Tuple[torch.Tensor, ...]


class _EncoderBase(torch.nn.Module):
    r"""This abstract class serves as a base for the 3 `Encoder` abstractions
    in AllenNLP.
    - [`Seq2SeqEncoders`](./seq2seq_encoders/seq2seq_encoder.md)
    - [`Seq2VecEncoders`](./seq2vec_encoders/seq2vec_encoder.md)

    Additionally, this class provides functionality for sorting sequences by
    length so they can be consumed by Pytorch RNN classes, which require their
    inputs to be sorted by length. Finally, it also provides optional
    statefulness to all of it's subclasses by allowing the caching and
    retrieving of the hidden states of RNNs.
    """

    def __init__(self, stateful: bool = False) -> None:
        super().__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(
        self,
        module: Callable[
            [PackedSequence, Optional[RnnState]],
            Tuple[Union[PackedSequence, torch.Tensor], RnnState],
        ],
        inputs: torch.Tensor,
        mask: torch.Tensor,
        hidden_state: Optional[RnnState] = None,
    ):
        r"""This function exists because Pytorch RNNs require that their inputs
        be sorted before being passed as input. As all of our Seq2xxxEncoders
        use this functionality, it is provided in a base class. This method can
        be called on any module which takes as input a `PackedSequence` and
        some `hidden_state`, which can either be a tuple of tensors or a tensor.

        As all of our Seq2xxxEncoders have different return types, we return
        `sorted` outputs from the module, which is called directly.
        Additionally, we return the indices into the batch dimension required
        to restore the tensor to it's correct, unsorted order and the number of
        valid batch elements (i.e the number of elements in the batch which are
        not completely masked). This un-sorting and re-padding of the module
        outputs is left to the subclasses because their outputs have different
        types and handling them smoothly here is difficult.

        # Parameters

        module : `Callable[[PackedSequence, Optional[RnnState]],
                            Tuple[Union[PackedSequence, torch.Tensor],
                            RnnState]]`, required.
            A function to run on the inputs. In most cases, this is a
            `torch.nn.Module`.
        inputs : `torch.Tensor`, required.
            A tensor of shape `(batch_size, sequence_length, embedding_size)`
            representing the inputs to the Encoder.
        mask : `torch.Tensor`, required.
            A tensor of shape `(batch_size, sequence_length)`, representing
            masked and non-masked elements of the sequence for each element in
            the batch.
        hidden_state : `Optional[RnnState]`, (default = None).
            A single tensor of shape (num_layers, batch_size, hidden_size)
            representing the state of an RNN with or a tuple of tensors of
            shapes (num_layers, batch_size, hidden_size) and
            (num_layers, batch_size, memory_size), representing the hidden
            state and memory state of an LSTM-like RNN.

        # Returns

        module_output : `Union[torch.Tensor, PackedSequence]`.
            A Tensor or PackedSequence representing the output of the Pytorch
            Module. The batch size dimension will be equal to `num_valid`, as
            sequences  of zero length are clipped off before the module is
            called, as Pytorch cannot handle zero length sequences.
        final_states : `Optional[RnnState]`
            A Tensor representing the hidden state of the Pytorch Module. This
            can either be a single tensor of shape (num_layers, num_valid,
            hidden_size), for instance in the case of a GRU, or a tuple of
            tensors, such as those required for an LSTM.
        restoration_indices : `torch.LongTensor`
            A tensor of shape `(batch_size,)`, describing the re-indexing
            required to transform the outputs back to their original batch
            order.
        """
        # In some circumstances you may have sequences of zero length.
        # `pack_padded_sequence` requires all sequence lengths to be > 0, so
        # remove sequences of zero length before calling self._module, then
        # fill with zeros.

        # First count how many sequences are empty.
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item()

        sequence_lengths = mask.long().sum(-1)
        (
            sorted_inputs,
            sorted_sequence_lengths,
            restoration_indices,
            sorting_indices,
        ) = sort_batch_by_length(inputs, sequence_lengths)

        # Now create a PackedSequence with only the non-empty, sorted sequences.
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs[:num_valid, :, :],
            sorted_sequence_lengths[:num_valid].data.tolist(),
            batch_first=True,
        )
        # Prepare the initial states.
        if not self.stateful:
            if hidden_state is None:
                initial_states: Any = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [
                    state.index_select(
                        1, sorting_indices)[:, :num_valid, :].contiguous()
                    for state in hidden_state
                ]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[
                    :, :num_valid, :].contiguous()

        else:
            initial_states = self._get_initial_states(batch_size, num_valid,
                                                      sorting_indices)

        # Actually call the module on the sorted PackedSequence.
        module_output, final_states = module(packed_sequence_input,
                                             initial_states)

        return module_output, final_states, restoration_indices

    def _get_initial_states(
        self, batch_size: int, num_valid: int, sorting_indices: torch.LongTensor
    ) -> Optional[RnnState]:
        r"""Returns an initial state for use in an RNN. Additionally, this
        method handles the batch size changing across calls by mutating the
        state to append initial states for new elements in the batch. Finally,
        it also handles sorting the states with respect to the sequence lengths
        of elements in the batch and removing rows which are completely padded.
        Importantly, this `mutates` the state if the current batch size is
        larger than when it was previously called.

        # Parameters

        batch_size : `int`, required.
            The batch size can change size across calls to stateful RNNs, so we
            need to know if we need to expand or shrink the states before
            returning them. Expanded states will be set to zero.
        num_valid : `int`, required.
            The batch may contain completely padded sequences which get removed
            before the sequence is passed through the encoder. We also need to
            clip these off of the state too.
        sorting_indices `torch.LongTensor`, required.
            Pytorch RNNs take sequences sorted by length. When we return the
            states to be used for a given call to `module.forward`, we need the
            states to match up to the sorted sequences, so before returning
            them, we sort the states using the same indices used to sort the
            sequences.

        # Returns

        This method has a complex return type because it has to deal with the
        first time it is called, when it has no state, and the fact that types
        of RNN have heterogeneous states.

        If it is the first time the module has been called, it returns `None`,
        regardless of the type of the `Module`.

        Otherwise, for LSTMs, it returns a tuple of `torch.Tensors` with shape
        `(num_layers, num_valid, state_size)` and `(num_layers, num_valid,
        memory_size)` respectively, or for GRUs, it returns a single
        `torch.Tensor` of shape `(num_layers, num_valid, state_size)`.
        """
        # We don't know the state sizes the first time calling forward,
        # so we let the module define what it's initial hidden state looks like.
        if self._states is None:
            return None

        # Otherwise, we have some previous states.
        if batch_size > self._states[0].size(1):
            # This batch is larger than the all previous states.
            # If so, resize the states.
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            # state has shape (num_layers, batch_size, hidden_size)
            for state in self._states:
                # This _must_ be inside the loop because some
                # RNNs have states with different last dimension sizes.
                zeros = state.new_zeros(state.size(0), num_states_to_concat,
                                        state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states

        elif batch_size < self._states[0].size(1):
            # This batch is smaller than the previous one.
            correctly_shaped_states = tuple(state[:, :batch_size, :] for state
                                            in self._states)
        else:
            correctly_shaped_states = self._states

        # At this point, our states are of shape (num_layers, batch_size,
        # hidden_size). However, the encoder uses sorted sequences and
        # additionally removes elements of the batch which are fully padded.
        # We need the states to match up to these sorted and filtered
        # sequences, so we do that in the next two blocks before returning the
        # state/s.
        if len(self._states) == 1:
            # GRUs only have a single state. This `unpacks` it from the
            # tuple and returns the tensor directly.
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(
                1, sorting_indices)
            return sorted_state[:, :num_valid, :].contiguous()
        else:
            # LSTMs have a state tuple of (state, memory).
            sorted_states = [
                state.index_select(1, sorting_indices) for state in
                correctly_shaped_states
            ]
            return tuple(state[:, :num_valid, :].contiguous()  # type: ignore
                         for state in sorted_states)

    def _update_states(self, final_states: RnnStateStorage,
                       restoration_indices: torch.LongTensor) -> None:
        r"""After the RNN has run forward, the states need to be updated.
        This method just sets the state to the updated new state, performing
        several pieces of book-keeping along the way - namely, unsorting the
        states and ensuring that the states of completely padded sequences are
        not updated. Finally, it also detaches the state variable from the
        computational graph, such that the graph can be garbage collected after
        each batch iteration.

        # Parameters

        final_states : `RnnStateStorage`, required.
            The hidden states returned as output from the RNN.
        restoration_indices : `torch.LongTensor`, required.
            The indices that invert the sorting used in `sort_and_run_forward`
            to order the states with respect to the lengths of the sequences in
            the batch.
        """
        # TODO(Mark): seems weird to sort here, but append zeros in the
        #  subclasses.
        # which way around is best?
        new_unsorted_states = [state.index_select(1, restoration_indices) for
                               state in final_states]

        if self._states is None:
            # We don't already have states, so just set the
            # ones we receive to be the current state.
            self._states = tuple(state.data for state in new_unsorted_states)
        else:
            # Now we've sorted the states back so that they correspond to the
            # original indices, we need to figure out what states we need to
            # update, because if we didn't use a state for a particular row,
            # we want to preserve its state. Thankfully, the rows which are
            # all zero in the state correspond exactly to those which aren't
            # used, so we create masks of shape (new_batch_size,), denoting
            # which states were used in the RNN computation.
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            # Masks for the unused states of shape (1, new_batch_size, 1)
            used_new_rows_mask = [
                (state[0, :, :].sum(-1) != 0.0).float().view(
                    1, new_state_batch_size, 1)
                for state in new_unsorted_states
            ]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                # The new state is smaller than the old one,
                # so just update the indices which we used.
                for old_state, new_state, used_mask in zip(
                    self._states, new_unsorted_states, used_new_rows_mask
                ):
                    # zero out all rows in the previous state
                    # which _were_ used in the current state.
                    masked_old_state = \
                        old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    # The old state is larger, so update the relevant parts of
                    # it.
                    old_state[:, :new_state_batch_size, :] = \
                        new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                # The states are the same size, so we just have to
                # deal with the possibility that some rows weren't used.
                new_states = []
                for old_state, new_state, used_mask in zip(
                    self._states, new_unsorted_states, used_new_rows_mask
                ):
                    # zero out all rows which _were_ used in the current state.
                    masked_old_state = old_state * (1 - used_mask)
                    # The old state is larger, so update the relevant parts of
                    # it.
                    new_state += masked_old_state
                    new_states.append(new_state.detach())

            # It looks like there should be another case handled here - when
            # the current_state_batch_size < new_state_batch_size. However,
            # this never happens, because the states themeselves are mutated
            # by appending zeros when calling _get_inital_states, meaning that
            # the new states are either of equal size, or smaller, in the case
            # that there are some unused elements (zero-length) for the RNN
            # computation.
            self._states = tuple(new_states)

    def reset_states(self, mask: Optional[torch.Tensor] = None) -> None:
        r"""Resets the internal states of a stateful encoder.

        # Parameters

        mask : `torch.Tensor`, optional.
            A tensor of shape `(batch_size,)` indicating which states should
            be reset. If not provided, all states will be reset.
        """
        if mask is None:
            self._states = None
        else:
            # state has shape (num_layers, batch_size, hidden_size). We reshape
            # mask to have shape (1, batch_size, 1) so that operations
            # broadcast properly.
            mask_batch_size = mask.size(0)
            mask = mask.float().view(1, mask_batch_size, 1)
            new_states = []
            assert self._states is not None
            for old_state in self._states:
                old_state_batch_size = old_state.size(1)
                if old_state_batch_size != mask_batch_size:
                    raise ValueError(
                        f"Trying to reset states using mask with incorrect "
                        f"batch size. "
                        f"Expected batch size: {old_state_batch_size}. "
                        f"Provided batch size: {mask_batch_size}."
                    )
                new_state = (1 - mask) * old_state
                new_states.append(new_state.detach())
            self._states = tuple(new_states)


class ElmoLstm(_EncoderBase):
    r"""A stacked, bidirectional LSTM which uses
    [`LstmCellWithProjection`'s](./lstm_cell_with_projection.md)
    with highway layers between the inputs to layers.
    The inputs to the forward and backward directions are independent -
    forward and backward states are not concatenated between layers.

    Additionally, this LSTM maintains its `own` state, which is updated every
    time `forward` is called. It is dynamically resized for different batch
    sizes and is designed for use with non-continuous inputs (i.e inputs which
    aren't formatted as a stream, such as text used for a language modeling
    task, which is how stateful RNNs are typically used).
    This is non-standard, but can be thought of as having an "end of sentence"
    state, which is carried across different sentences.

    # Parameters

    input_size : `int`, required
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required
        The dimension of the outputs of the LSTM.
    cell_size : `int`, required.
        The dimension of the memory cell of the `LstmCellWithProjection`.
    num_layers : `int`, required
        The number of bidirectional LSTMs to use.
    requires_grad : `bool`, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    recurrent_dropout_probability : `float`, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        [A Theoretically Grounded Application of Dropout in Recurrent Neural
        Networks](https://arxiv.org/abs/1512.05287).
    state_projection_clip_value : `float`, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value : `float`, optional, (default = None)
        The magnitude with which to clip the memory cell.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cell_size: int,
        num_layers: int,
        requires_grad: bool = False,
        recurrent_dropout_probability: float = 0.0,
        memory_cell_clip_value: Optional[float] = None,
        state_projection_clip_value: Optional[float] = None,
    ) -> None:
        super().__init__(stateful=True)

        # Required to be wrapped with a `PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        self.requires_grad = requires_grad

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(
                lstm_input_size,
                hidden_size,
                cell_size,
                go_forward,
                recurrent_dropout_probability,
                memory_cell_clip_value,
                state_projection_clip_value,
            )
            backward_layer = LstmCellWithProjection(
                lstm_input_size,
                hidden_size,
                cell_size,
                not go_forward,
                recurrent_dropout_probability,
                memory_cell_clip_value,
                state_projection_clip_value,
            )
            lstm_input_size = hidden_size

            self.add_module("forward_layer_{}".format(layer_index),
                            forward_layer)
            self.add_module("backward_layer_{}".format(layer_index),
                            backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self, inputs: torch.Tensor,  # type: ignore
                mask: torch.LongTensor) -> torch.Tensor:
        r"""Encodes the inputs.

        # Parameters

        inputs : `torch.Tensor`, required.
            A Tensor of shape `(batch_size, sequence_length, hidden_size)`.
        mask : `torch.LongTensor`, required.
            A binary mask of shape `(batch_size, sequence_length)` representing
            the non-padded elements in each sequence in the batch.

        # Returns

        A `torch.Tensor` of shape (num_layers, batch_size, sequence_length,
        hidden_size), where the num_layers dimension represents the LSTM output
        from that layer.
        """
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask)

        num_layers, num_valid, returned_timesteps, encoder_dim = \
            stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to
        # sort_and_run_forward.
        if num_valid < batch_size:
            zeros = stacked_sequence_output.new_zeros(
                num_layers, batch_size - num_valid, returned_timesteps,
                encoder_dim
            )
            stacked_sequence_output = torch.cat(
                [stacked_sequence_output, zeros], 1)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.new_zeros(num_layers, batch_size - num_valid,
                                        state_dim)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer
        # than the max length of the sequence to a Seq2StackEncoder. However,
        # packing and unpacking the sequences mean that the returned tensor
        # won't include these dimensions, because the RNN did not need to
        # process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.new_zeros(
                num_layers,
                batch_size,
                sequence_length_difference,
                stacked_sequence_output[0].size(-1),
            )
            stacked_sequence_output = torch.cat(
                [stacked_sequence_output, zeros], 2)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        return stacked_sequence_output.index_select(1, restoration_indices)

    def _lstm_forward(
        self,
        inputs: PackedSequence,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Encodes the inputs.

        # Parameters

        inputs : `PackedSequence`, required.
            A batch first `PackedSequence` to run the stacked LSTM over.
        initial_state : `Tuple[torch.Tensor, torch.Tensor]`, optional,
            (default = None)
            A tuple (state, memory) representing the initial hidden state and
            memory of the LSTM, with shape
            (num_layers, batch_size, 2 * hidden_size) and
            (num_layers, batch_size, 2 * cell_size) respectively.

        # Returns

        output_sequence : `torch.FloatTensor`
            The encoded sequence of shape
            (num_layers, batch_size, sequence_length, hidden_size)
        final_states : `Tuple[torch.FloatTensor, torch.FloatTensor]`
            The per-layer final (state, memory) states of the LSTM, with shape
            (num_layers, batch_size, 2 * hidden_size) and
            (num_layers, batch_size, 2 * cell_size)
            respectively. The last dimension is duplicated because it
            contains the state/memory for both the forward and backward layers.
        """
        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = \
                [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise ConfigurationError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))

        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, "forward_layer_{}".format(
                layer_index))
            backward_layer = getattr(self, "backward_layer_{}".format(
                layer_index))

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(
                    self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(
                    self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None  # type: ignore
                backward_state = None  # type: ignore

            forward_output_sequence, forward_state = forward_layer(
                forward_output_sequence, batch_lengths, forward_state
            )
            backward_output_sequence, backward_state = backward_layer(
                backward_output_sequence, batch_lengths, backward_state
            )
            # Skip connections, just adding the input to the output.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(
                torch.cat([forward_output_sequence, backward_output_sequence],
                          -1)
            )
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append(
                (
                    torch.cat([forward_state[0], backward_state[0]], -1),
                    torch.cat([forward_state[1], backward_state[1]], -1),
                )
            )

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(
            sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of
        # shape (num_layers, batch_size, hidden_size) and
        # (num_layers, batch_size, cell_size) respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor, torch.FloatTensor] = (
            torch.cat(final_hidden_states, 0),
            torch.cat(final_memory_states, 0),
        )
        return stacked_sequence_outputs, final_state_tuple

    def load_weights(self, weight_file: str) -> None:
        r"""Load the pre-trained weights from the file.
        """
        requires_grad = self.requires_grad

        with h5py.File(weight_file, "r") as fin:
            for i_layer, lstms in enumerate(zip(self.forward_layers,
                                                self.backward_layers)):
                for j_direction, lstm in enumerate(lstms):
                    # lstm is an instance of LSTMCellWithProjection
                    cell_size = lstm.cell_size

                    dataset = fin["RNN_%s" % j_direction]["RNN"][
                        "MultiRNNCell"][
                        "Cell%s" % i_layer
                    ]["LSTMCell"]

                    # tensorflow packs together both W and U matrices into one
                    # matrix, but pytorch maintains individual matrices.  In
                    # addition, tensorflow packs the gates as input, memory,
                    # forget, output but pytorch uses input, forget, memory,
                    # output.  So we need to modify the weights.
                    tf_weights = numpy.transpose(dataset["W_0"][...])
                    torch_weights = tf_weights.copy()

                    # split the W from U matrices
                    input_size = lstm.input_size
                    input_weights = torch_weights[:, :input_size]
                    recurrent_weights = torch_weights[:, input_size:]
                    tf_input_weights = tf_weights[:, :input_size]
                    tf_recurrent_weights = tf_weights[:, input_size:]

                    # handle the different gate order convention
                    for torch_w, tf_w in [
                        [input_weights, tf_input_weights],
                        [recurrent_weights, tf_recurrent_weights],
                    ]:
                        torch_w[(1 * cell_size): (2 * cell_size), :] = tf_w[
                            (2 * cell_size): (3 * cell_size), :
                        ]
                        torch_w[(2 * cell_size): (3 * cell_size), :] = tf_w[
                            (1 * cell_size): (2 * cell_size), :
                        ]

                    lstm.input_linearity.weight.data.copy_(torch.FloatTensor(
                        input_weights))
                    lstm.state_linearity.weight.data.copy_(torch.FloatTensor(
                        recurrent_weights))
                    lstm.input_linearity.weight.requires_grad = requires_grad
                    lstm.state_linearity.weight.requires_grad = requires_grad

                    # the bias weights
                    tf_bias = dataset["B"][...]
                    # tensorflow adds 1.0 to forget gate bias instead of
                    # modifying the parameters...
                    tf_bias[(2 * cell_size): (3 * cell_size)] += 1
                    torch_bias = tf_bias.copy()
                    torch_bias[(1 * cell_size): (2 * cell_size)] = tf_bias[
                        (2 * cell_size): (3 * cell_size)
                    ]
                    torch_bias[(2 * cell_size): (3 * cell_size)] = tf_bias[
                        (1 * cell_size): (2 * cell_size)
                    ]
                    lstm.state_linearity.bias.data.copy_(torch.FloatTensor(
                        torch_bias))
                    lstm.state_linearity.bias.requires_grad = requires_grad

                    # the projection weights
                    proj_weights = numpy.transpose(dataset["W_P_0"][...])
                    lstm.state_projection.weight.data.copy_(torch.FloatTensor(
                        proj_weights))
                    lstm.state_projection.weight.requires_grad = requires_grad


class LstmCellWithProjection(torch.nn.Module):
    r"""An LSTM with Recurrent Dropout and a projected and clipped hidden state
    and memory. Note: this implementation is slower than the native Pytorch
    LSTM because it cannot make use of CUDNN optimizations for stacked RNNs due
    to and variational dropout and the custom nature of the cell state.

    # Parameters

    input_size : `int`, required.
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required.
        The dimension of the outputs of the LSTM.
    cell_size : `int`, required.
        The dimension of the memory cell used for the LSTM.
    go_forward : `bool`, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability : `float`, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        [A Theoretically Grounded Application of Dropout in Recurrent Neural
        Networks] (https://arxiv.org/abs/1512.05287). Implementation wise,
        this simply applies a fixed dropout mask per sequence to the recurrent
        connection of the LSTM.
    state_projection_clip_value : `float`, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value : `float`, optional, (default = None)
        The magnitude with which to clip the memory cell.

    # Returns

    output_accumulator : `torch.FloatTensor`
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    final_state : `Tuple[torch.FloatTensor, torch.FloatTensor]`
        The final (state, memory) states of the LSTM, with shape
        (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
        respectively. The first dimension is 1 in order to match the Pytorch
        API for returning stacked LSTM states.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cell_size: int,
        go_forward: bool = True,
        recurrent_dropout_probability: float = 0.0,
        memory_cell_clip_value: Optional[float] = None,
        state_projection_clip_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        # Required to be wrapped with a `PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability

        # We do the projections for all the gates all at once.
        self.input_linearity = torch.nn.Linear(
            input_size, 4 * cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(
            hidden_size, 4 * cell_size, bias=True)

        # Additional projection matrix for making the hidden state smaller.
        self.state_projection = torch.nn.Linear(
            cell_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data,
                         [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data,
                         [self.cell_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.cell_size:
                                       2 * self.cell_size].fill_(1.0)

    def forward(  # type: ignore
        self,
        inputs: torch.FloatTensor,
        batch_lengths: List[int],
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        r"""Process the inputs.

        # Parameters

        inputs : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        batch_lengths : `List[int]`, required.
            A list of length batch_size containing the lengths of the sequences
            in batch.
        initial_state : `Tuple[torch.Tensor, torch.Tensor]`, optional,
            (default = None)
            A tuple (state, memory) representing the initial hidden state and
            memory of the LSTM. The `state` has shape (1, batch_size,
            hidden_size) and the `memory` has shape (1, batch_size, cell_size).

        # Returns

        output_accumulator : `torch.FloatTensor`
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
        final_state : `Tuple[`torch.FloatTensor, torch.FloatTensor]`
            A tuple (state, memory) representing the initial hidden state and
            memory of the LSTM. The `state` has shape (1, batch_size,
            hidden_size) and the `memory` has shape (1, batch_size, cell_size).
        """
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]

        output_accumulator = inputs.new_zeros(batch_size, total_timesteps,
                                              self.hidden_size)

        if initial_state is None:
            full_batch_previous_memory = inputs.new_zeros(batch_size,
                                                          self.cell_size)
            full_batch_previous_state = inputs.new_zeros(batch_size,
                                                         self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(
                self.recurrent_dropout_probability, full_batch_previous_state
            )
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else \
                total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch
            # dimension which we need to use for this timestep, because the
            # sequences have variable length, so once the index is greater than
            # the length of this particular batch sequence, we no longer need
            # to do the computation for this sequence. The key thing to
            # recognise here is that the batch inputs must be _ordered_ by
            # length from longest (first in batch) to shortest (last) so
            # initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of
                # elements in the batch?
                # Second conditional: Does the next shortest sequence beyond
                # the current batch index require computation use this timestep?
                while (
                    current_length_index < (len(batch_lengths) - 1)
                    and batch_lengths[current_length_index + 1] > index
                ):
                    current_length_index += 1

            # Actually get the slices of the batch which we
            # need for the computation at this timestep.
            # shape (batch_size, cell_size)
            previous_memory = \
                full_batch_previous_memory[0: current_length_index + 1].clone()
            # Shape (batch_size, hidden_size)
            previous_state = \
                full_batch_previous_state[0: current_length_index + 1].clone()
            # Shape (batch_size, input_size)
            timestep_input = inputs[0: current_length_index + 1, index]

            # Do the projections for all the gates all at once.
            # Both have shape (batch_size, 4 * cell_size)
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            input_gate = torch.sigmoid(
                projected_input[:, (0 * self.cell_size): (1 * self.cell_size)]
                + projected_state[:, (0 * self.cell_size): (1 * self.cell_size)]
            )
            forget_gate = torch.sigmoid(
                projected_input[:, (1 * self.cell_size): (2 * self.cell_size)]
                + projected_state[:, (1 * self.cell_size): (2 * self.cell_size)]
            )
            memory_init = torch.tanh(
                projected_input[:, (2 * self.cell_size): (3 * self.cell_size)]
                + projected_state[:, (2 * self.cell_size): (3 * self.cell_size)]
            )
            output_gate = torch.sigmoid(
                projected_input[:, (3 * self.cell_size): (4 * self.cell_size)]
                + projected_state[:, (3 * self.cell_size): (4 * self.cell_size)]
            )
            memory = input_gate * memory_init + forget_gate * previous_memory

            # Here is the non-standard part of this LSTM cell; first, we clip
            # the memory cell, then we project the output of the timestep to a
            # smaller size and again clip it.

            if self.memory_cell_clip_value:

                memory = torch.clamp(
                    memory, -self.memory_cell_clip_value,
                    self.memory_cell_clip_value
                )

            # shape (current_length_index, cell_size)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)

            # shape (current_length_index, hidden_size)
            timestep_output = self.state_projection(
                pre_projection_timestep_output)
            if self.state_projection_clip_value:

                timestep_output = torch.clamp(
                    timestep_output,
                    -self.state_projection_clip_value,
                    self.state_projection_clip_value,
                )

            # Only do dropout if the dropout prob is > 0.0 and we are in
            # training mode.
            if dropout_mask is not None:
                timestep_output = \
                    timestep_output * dropout_mask[0: current_length_index + 1]

            # We've been doing computation with less than the full batch, so
            # here we create a new variable for the the whole batch at this
            # timestep and insert the result for the relevant elements of the
            # batch into it.
            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0: current_length_index + 1] = memory
            full_batch_previous_state[0: current_length_index + 1] = \
                timestep_output
            output_accumulator[0: current_length_index + 1, index] = \
                timestep_output

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, ...). As this
        # LSTM cell cannot be stacked, the first dimension here is just 1.
        final_state = (
            full_batch_previous_state.unsqueeze(0),
            full_batch_previous_memory.unsqueeze(0),
        )

        return output_accumulator, final_state


class Highway(torch.nn.Module):
    r"""A [Highway layer](https://arxiv.org/abs/1505.00387) does a gated
    combination of a linear transformation and a non-linear transformation of
    its input. :math:`y = g * x + (1 - g) * f(A(x))`,
    where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed
    as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input,
    returning the final result.

    # Parameters

    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape
        `(batch_size, ..., input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    activation : `Callable[[torch.Tensor], torch.Tensor]`, optional
        (default=`torch.nn.functional.relu`)
        The non-linearity to use in the highway layers.
    """

    def __init__(self, input_dim: int, num_layers: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] =
                 torch.nn.functional.relu,) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2)
             for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.
            # We do that by setting the bias on `B(x)` to be positive, because
            # that means `g` will be biased to be high, so we will carry the
            # input forward.  The bias on `B(x)` is the second half of the
            # bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)  # type: ignore

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify
            # the initialization above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class Embedding(torch.nn.Module):
    r"""A more featureful embedding module than the default in Pytorch.  Adds
    the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only
           makes sense with non-trainable embeddings).
        5. build all of this easily `from_params`

    Note that if you are using our data API and are trying to embed a
    [`TextField`](../../data/fields/text_field.md), you should use a
    [`TextFieldEmbedder`](../text_field_embedders/text_field_embedder.md)
    instead of using this directly.

    # Parameters

    num_embeddings : `int`
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : `int`
        The size of each embedding vector.
    projection_dim : `int`, (optional, default=None)
        If given, we add a projection layer after the embedding layer.  This
        really only makes sense if `trainable` is `False`.
    weight : `torch.FloatTensor`, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the
        use of pretrained vectors.
    padding_index : `int`, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : `bool`, (optional, default=True)
        Whether or not to optimize the embedding parameters.
    max_norm : `float`, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser
        than this
    norm_type : `float`, (optional, default=2)
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : `bool`, (optional, default=False)
        If given, this will scale gradients by the frequency of the words in
        the mini-batch.
    sparse : `bool`, (optional, default=False)
        Whether or not the Pytorch backend should use a sparse representation
        of the embedding weight.
    vocab_namespace : `str`, (optional, default=None)
        In case of fine-tuning/transfer learning, the model's embedding matrix
        needs to be extended according to the size of extended-vocabulary. To
        be able to know how much to extend the embedding-matrix, it's necessary
        to know which vocab_namspace was used to construct it in the original
        training. We store vocab_namespace used during the original training as
        an attribute, so that it can be retrieved during fine-tuning.
    pretrained_file : `str`, (optional, default=None)
        Used to keep track of what is the source of the weights and loading
        more embeddings at test time. **It does not load the weights from this
        pretrained_file.** For that purpose, use `Embedding.from_params`.

    # Returns

    An Embedding module.
    """

    default_implementation = "embedding"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        projection_dim: Optional[int] = None,
        weight: Optional[torch.FloatTensor] = None,
        padding_index: Optional[int] = None,
        trainable: bool = True,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        vocab_namespace: Optional[str] = None,
        pretrained_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = vocab_namespace
        self._pretrained_file = pretrained_file

        self.output_dim = projection_dim or embedding_dim

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise ConfigurationError(
                    "A weight matrix was passed with contradictory embedding "
                    "shapes."
                )
            self.weight = torch.nn.Parameter(weight,
                                             requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None  # type: ignore

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # type: ignore
        # tokens may have extra dimensions
        # (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass tokens to
        # util.combine_initial_dims (which is a no-op if there are no extra
        # dimensions). Remember the original size.
        original_size = tokens.size()
        tokens = combine_initial_dims(tokens)

        embedded = embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)  # type: ignore
            embedded = projection(embedded)
        return embedded


class TimeDistributed(torch.nn.Module):
    r"""Given an input shaped like `(batch_size, time_steps, [rest])` and a
    `Module` that takes inputs like `(batch_size, [rest])`, `TimeDistributed`
    reshapes the input to be `(batch_size * time_steps, [rest])`, applies the
    contained `Module`, then reshapes it back.

    Note that while the above gives shapes with `batch_size` first, this
    `Module` also works if `batch_size` is second - we always just combine the
    first two dimensions, then split them.

    It also reshapes keyword arguments unless they are not tensors or their
    name is specified in the optional `pass_through` iterable.
    """

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs,
                pass_through: Optional[List[str]] = None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [self._reshape_tensor(input_tensor)
                           for input_tensor in inputs]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        new_size = some_input.size()[:2] + reshaped_outputs.size()[1:]
        outputs = reshaped_outputs.contiguous().view(new_size)

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)


def add_sentence_boundary_token_ids(
        tensor: torch.Tensor, mask: torch.Tensor,
        sentence_begin_token: Any, sentence_end_token: Any) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    r"""Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size `(batch_size, timesteps)` or
    `(batch_size, timesteps, dim)` this returns a tensor of shape
    `(batch_size, timesteps + 2)` or `(batch_size, timesteps + 2, dim)`
    respectively.

    Returns both the new tensor and updated mask.

    # Parameters

    tensor : `torch.Tensor`
        A tensor of shape `(batch_size, timesteps)` or
        `(batch_size, timesteps, dim)`
    mask : `torch.Tensor`
         A tensor of shape `(batch_size, timesteps)`
    sentence_begin_token: Any (anything that can be broadcast in torch for
        assignment)
        For 2D input, a scalar with the <S> id. For 3D input, a tensor with
        length dim.
    sentence_end_token: Any (anything that can be broadcast in torch for
        assignment)
        For 2D input, a scalar with the </S> id. For 3D input, a tensor with
        length dim.

    # Returns

    tensor_with_boundary_tokens : `torch.Tensor`
        The tensor with the appended and prepended boundary tokens. If the
        input was 2D, it has shape (batch_size, timesteps + 2) and if the
        input was 3D, it has shape (batch_size, timesteps + 2, dim).
    new_mask : `torch.Tensor`
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    # TODO: matthewp, profile this transfer
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens != 0).long()
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
        new_mask = (
                (tensor_with_boundary_tokens > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError(
            "add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask


def get_device_of(tensor: torch.Tensor) -> int:
    r"""Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def remove_sentence_boundaries(tensor: torch.Tensor, mask: torch.Tensor) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    r"""Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size `(batch_size, timesteps, dim)`
    this returns a tensor of shape `(batch_size, timesteps - 2, dim)` after
    removing the beginning and end sentence markers.  The sentences are
    assumed to be padded on the right, with the beginning of each sentence
    assumed to occur at index 0 (i.e., `mask[:, 0]` is assumed to be 1).

    Returns both the new tensor and updated mask.

    This function is the inverse of `add_sentence_boundary_token_ids`.

    # Parameters

    tensor : `torch.Tensor`
        A tensor of shape `(batch_size, timesteps, dim)`
    mask : `torch.Tensor`
         A tensor of shape `(batch_size, timesteps)`

    # Returns

    tensor_without_boundary_tokens : `torch.Tensor`
        The tensor after removing the boundary tokens of shape
        `(batch_size, timesteps - 2, dim)`
    new_mask : `torch.Tensor`
        The new mask for the tensor of shape `(batch_size, timesteps - 2)`.
    """
    # TODO: matthewp, profile this transfer
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.long)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, : (j - 2), :] = \
                tensor[i, 1: (j - 1), :]
            new_mask[i, : (j - 2)] = 1

    return tensor_without_boundary_tokens, new_mask


class ConfigurationError(Exception):
    r"""The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        # TODO(brendanr): Is there some reason why we need repr here? It
        # produces horrible output for simple multi-line error messages.
        return self.message


def block_orthogonal(tensor: torch.Tensor, split_sizes: List[int],
                     gain: float = 1.0) -> None:
    r"""An initializer which allows initializing model parameters in "blocks".
    This is helpful in the case of recurrent models which use multiple gates
    applied to linear projections, which can be computed efficiently if they
    are concatenated together. However, they are separate parameters which
    should be initialized independently.

    # Parameters

    tensor : `torch.Tensor`, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length `tensor.ndim()` specifying the size of the
        blocks along that particular dimension. E.g. `[10, 20]` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any(a % b != 0 for a, b in zip(sizes, split_sizes)):
        raise ConfigurationError(
            "tensor dimensions must be divisible by their respective "
            "split_sizes. Found size: {} and split_sizes: {}".format(
                sizes, split_sizes)
        )
    indexes = [list(range(0, max_size, split)) for max_size, split in zip(
        sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is required because we
        # could have an arbitrary number of dimensions. The actual slices we
        # need are the start_index: start_index + step for each dimension in
        # the tensor.
        block_slice = tuple(
            slice(start_index, start_index + step) for start_index, step in
            index_and_step_tuples
        )
        data[block_slice] = torch.nn.init.orthogonal_(
            tensor[block_slice].contiguous(), gain=gain)


def get_dropout_mask(dropout_probability: float,
                     tensor_for_masking: torch.Tensor):
    r"""Computes and returns an element-wise dropout mask for a given tensor,
    where each element in the mask is dropped out with probability
    dropout_probability. Note that the mask is NOT applied to the tensor -
    the tensor is passed to retain the correct CUDA tensor type for the mask.

    # Parameters

    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Tensor, required.

    # Returns

    A torch.FloatTensor consisting of the binary mask scaled by
    1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of
    applying this mask and the original tensor are the same.
    """
    binary_mask = (
            torch.rand(tensor_for_masking.size()) > dropout_probability).to(
        tensor_for_masking.device
    )
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def combine_initial_dims(tensor: torch.Tensor) -> torch.Tensor:
    r"""Given a (possibly higher order) tensor of ids with shape
    (d1, ..., dn, sequence_length) Return a view that's
    (d1 * ... * dn, sequence_length). If original tensor is 1-d or 2-d,
    return it as is.
    """
    if tensor.dim() <= 2:
        return tensor
    else:
        return tensor.view(-1, tensor.size(-1))


def uncombine_initial_dims(tensor: torch.Tensor, original_size: torch.Size) -> \
        torch.Tensor:
    r"""Given a tensor of embeddings with shape
    (d1 * ... * dn, sequence_length, embedding_dim) and the original shape
    (d1, ..., dn, sequence_length), return the reshaped tensor of embeddings
    with shape (d1, ..., dn, sequence_length, embedding_dim).
    If original size is 1-d or 2-d, return it as is.
    """
    if len(original_size) <= 2:
        return tensor
    else:
        view_args = list(original_size) + [tensor.size(-1)]
        return tensor.view(*view_args)


class ScalarMix(torch.nn.Module):
    r"""Computes a parameterised scalar mixture of N tensors,
    `mixture = gamma * sum(s_k * tensor_k)` where `s = softmax(w)`, with `w`
    and `gamma` scalar parameters.

    In addition, if `do_layer_norm=True` then apply layer normalization to
    each tensor before weighting.
    """

    def __init__(self, mixture_size: int, do_layer_norm: bool = False,
                 initial_scalar_parameters: Optional[List[float]] = None,
                 trainable: bool = True,) -> None:
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ConfigurationError(
                "Length of initial_scalar_parameters {} differs "
                "from mixture_size {}".format(initial_scalar_parameters,
                                              mixture_size)
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]),
                    requires_grad=trainable
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]),
                               requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor],  # type: ignore
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Compute a weighted average of the `tensors`.  The input tensors an
        be any shape with at least two dimensions, but must all be the same
        shape.

        When `do_layer_norm=True`, the `mask` is required input.  If the
        `tensors` are dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the
        `mask` is dimensioned `(dim_0, ..., dim_{n-1})`, as in the typical
        case with `tensors` of shape `(batch_size, timesteps, dim)` and `mask`
        of shape `(batch_size, timesteps)`.

        When `do_layer_norm=False` the `mask` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ConfigurationError(
                "{} tensors were passed, but the module was initialized to "
                "mix {} tensors.".format(len(tensors), self.mixture_size)
            )

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2) /
                num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        # pylint: disable=unnecessary-comprehension
        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]),
            dim=0
        )
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            assert mask is not None
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(
                    weight * _do_layer_norm(tensor, broadcast_mask,
                                            num_elements_not_masked)
                )
            return self.gamma * sum(pieces)
