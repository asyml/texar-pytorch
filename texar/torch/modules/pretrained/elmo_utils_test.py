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
Unit tests for utils of ELMo modules.

Code adapted from:
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/common/util_test.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/modules/elmo_test.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/modules/encoder_base_test.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/modules/lstm_cell_with_projection_test.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/modules/highway_test.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/modules/time_distributed_test.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/nn/initializers_test.py`
    `https://github.com/allenai/allennlp/blob/master/allennlp/tests/nn/util_test.py`
"""

import unittest

import h5py
import json
import numpy
import tempfile
import torch

from numpy.testing import assert_array_almost_equal, assert_almost_equal
from torch.nn import LSTM, RNN, Embedding, Module, Parameter

from texar.torch.data.tokenizers.elmo_tokenizer_utils import batch_to_ids
from texar.torch.data.data_utils import maybe_download
from texar.torch.modules.pretrained.elmo_utils import (
    Highway, LstmCellWithProjection, _EncoderBase, _ElmoBiLm, TimeDistributed,
    remove_sentence_boundaries, add_sentence_boundary_token_ids,
    block_orthogonal, ConfigurationError, combine_initial_dims,
    uncombine_initial_dims, ScalarMix)
from texar.torch.utils.test import cuda_test
from texar.torch.utils.utils import sort_batch_by_length


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class TestElmoBiLm(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.options_file = maybe_download(
            'https://github.com/allenai/allennlp/blob/master/allennlp/tests/'
            'fixtures/elmo/options.json?raw=true',
            self.tmp_dir.name)
        self.weight_file = maybe_download(
            'https://github.com/allenai/allennlp/blob/master/allennlp/tests/'
            'fixtures/elmo/lm_weights.hdf5?raw=true',
            self.tmp_dir.name)
        self.sentences_json_file = maybe_download(
            'https://github.com/allenai/allennlp/blob/master/allennlp/tests/'
            'fixtures/elmo/sentences.json?raw=true',
            self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _load_sentences_embeddings(self):
        r"""Load the test sentences and the expected LM embeddings.

        These files loaded in this method were created with a batch-size of 3.
        Due to idiosyncrasies with TensorFlow, the 30 sentences in
        sentences.json are split into 3 files in which the k-th sentence in
        each is from batch k.

        This method returns a (sentences, embeddings) pair where each is a
        list of length batch_size. Each list contains a sublist with
        total_sentence_count / batch_size elements.  As with the original files,
        the k-th element in the sublist is in batch k.
        """
        with open(self.sentences_json_file) as fin:
            sentences = json.load(fin)

        # the expected embeddings
        expected_lm_embeddings = []
        for k in range(len(sentences)):
            embed_fname = maybe_download(
                'https://github.com/allenai/allennlp/blob/master/allennlp/'
                'tests/fixtures/elmo/lm_embeddings_{}.hdf5?raw=true'.format(k),
                self.tmp_dir.name)
            expected_lm_embeddings.append([])
            with h5py.File(embed_fname, "r") as fin:
                for i in range(10):
                    sent_embeds = fin["%s" % i][...]
                    sent_embeds_concat = numpy.concatenate(
                        (sent_embeds[0, :, :], sent_embeds[1, :, :]), axis=-1
                    )
                    expected_lm_embeddings[-1].append(sent_embeds_concat)

        return sentences, expected_lm_embeddings

    def test_elmo_bilm(self):
        # get the raw data
        sentences, expected_lm_embeddings = self._load_sentences_embeddings()

        # load the test model
        elmo_bilm = _ElmoBiLm(self.options_file, self.weight_file)

        batches = [[sentences[j][i].split() for j in range(3)]
                   for i in range(10)]

        # Now finally we can iterate through batches.
        for i, batch in enumerate(batches):
            lm_embeddings = elmo_bilm(batch_to_ids(batch[:3]))
            top_layer_embeddings, mask = remove_sentence_boundaries(
                lm_embeddings["activations"][2], lm_embeddings["mask"]
            )

            # check the mask lengths
            lengths = mask.data.numpy().sum(axis=1)
            batch_sentences = [sentences[k][i] for k in range(3)]
            expected_lengths = [len(sentence.split()) for sentence in
                                batch_sentences]
            self.assertEqual(lengths.tolist(), expected_lengths)

            # get the expected embeddings and compare!
            expected_top_layer = [expected_lm_embeddings[k][i] for k in
                                  range(3)]
            for k in range(3):
                self.assertTrue(
                    numpy.allclose(
                        top_layer_embeddings[k, : lengths[k], :].data.numpy(),
                        expected_top_layer[k],
                        atol=1.0e-6,
                    )
                )


class TestEncoderBase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.lstm = LSTM(
            bidirectional=True, num_layers=3, input_size=3, hidden_size=7,
            batch_first=True
        )
        self.rnn = RNN(
            bidirectional=True, num_layers=3, input_size=3, hidden_size=7,
            batch_first=True
        )
        self.encoder_base = _EncoderBase(stateful=True)

        tensor = torch.rand([5, 7, 3])
        tensor[1, 6:, :] = 0
        tensor[3, 2:, :] = 0
        self.tensor = tensor
        mask = torch.ones(5, 7)
        mask[1, 6:] = 0
        mask[2, :] = 0  # <= completely masked
        mask[3, 2:] = 0
        mask[4, :] = 0  # <= completely masked
        self.mask = mask

        self.batch_size = 5
        self.num_valid = 3
        sequence_lengths = mask.long().sum(-1)
        _, _, restoration_indices, sorting_indices = sort_batch_by_length(
            tensor, sequence_lengths)
        self.sorting_indices = sorting_indices
        self.restoration_indices = restoration_indices

    def test_non_stateful_states_are_sorted_correctly(self):
        encoder_base = _EncoderBase(stateful=False)
        initial_states = (torch.randn(6, 5, 7), torch.randn(6, 5, 7))
        # Check that we sort the state for non-stateful encoders. To test
        # we'll just use a "pass through" encoder, as we aren't actually testing
        # the functionality of the encoder here anyway.
        _, states, restoration_indices = encoder_base.sort_and_run_forward(
            lambda *x: x, self.tensor, self.mask, initial_states
        )
        # Our input tensor had 2 zero length sequences, so we need
        # to concat a tensor of shape
        # (num_layers * num_directions, batch_size - num_valid, hidden_dim),
        # to the output before unsorting it.
        zeros = torch.zeros([6, 2, 7])

        # sort_and_run_forward strips fully-padded instances from the batch;
        # in order to use the restoration_indices we need to add back the two
        # that got stripped. What we get back should match what we started with.
        for state, original in zip(states, initial_states):
            assert list(state.size()) == [6, 3, 7]
            state_with_zeros = torch.cat([state, zeros], 1)
            unsorted_state = state_with_zeros.index_select(1,
                                                           restoration_indices)
            for index in [0, 1, 3]:
                numpy.testing.assert_array_equal(
                    unsorted_state[:, index, :].data.numpy(),
                    original[:, index, :].data.numpy()
                )

    def test_get_initial_states(self):
        # First time we call it, there should be no state, so we should return
        # None.
        assert (
            self.encoder_base._get_initial_states(
                self.batch_size, self.num_valid, self.sorting_indices
            )
            is None
        )

        # First test the case that the previous state is _smaller_ than the
        # current state input.
        initial_states = (torch.randn([1, 3, 7]), torch.randn([1, 3, 7]))
        self.encoder_base._states = initial_states
        # sorting indices are: [0, 1, 3, 2, 4]
        returned_states = self.encoder_base._get_initial_states(
            self.batch_size, self.num_valid, self.sorting_indices
        )

        correct_expanded_states = [
            torch.cat([state, torch.zeros([1, 2, 7])], 1)
            for state in initial_states
        ]
        # State should have been expanded with zeros to have shape
        # (1, batch_size, hidden_size).
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0].data.numpy(),
            correct_expanded_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1].data.numpy(),
            correct_expanded_states[1].data.numpy()
        )

        # The returned states should be of shape (1, num_valid, hidden_size) and
        # they also should have been sorted with respect to the indices.
        # sorting indices are: [0, 1, 3, 2, 4]

        correct_returned_states = [
            state.index_select(1, self.sorting_indices)[:, : self.num_valid, :]
            for state in correct_expanded_states
        ]

        numpy.testing.assert_array_equal(
            returned_states[0].data.numpy(),
            correct_returned_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            returned_states[1].data.numpy(),
            correct_returned_states[1].data.numpy()
        )

        # Now test the case that the previous state is larger:
        original_states = (torch.randn([1, 10, 7]), torch.randn([1, 10, 7]))
        self.encoder_base._states = original_states
        # sorting indices are: [0, 1, 3, 2, 4]
        returned_states = self.encoder_base._get_initial_states(
            self.batch_size, self.num_valid, self.sorting_indices
        )
        # State should not have changed, as they were larger
        # than the batch size of the requested states.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0].data.numpy(),
            original_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1].data.numpy(),
            original_states[1].data.numpy()
        )

        # The returned states should be of shape (1, num_valid, hidden_size)
        # and they also should have been sorted with respect to the indices.
        correct_returned_state = [
            x.index_select(1, self.sorting_indices)[:, : self.num_valid, :]
            for x in original_states
        ]
        numpy.testing.assert_array_equal(
            returned_states[0].data.numpy(),
            correct_returned_state[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            returned_states[1].data.numpy(),
            correct_returned_state[1].data.numpy()
        )

    def test_update_states(self):
        assert self.encoder_base._states is None
        initial_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])

        index_selected_initial_states = (
            initial_states[0].index_select(1, self.restoration_indices),
            initial_states[1].index_select(1, self.restoration_indices),
        )

        self.encoder_base._update_states(initial_states,
                                         self.restoration_indices)
        # State was None, so the updated state should just be the sorted given
        # state.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0].data.numpy(),
            index_selected_initial_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1].data.numpy(),
            index_selected_initial_states[1].data.numpy()
        )

        new_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])
        # tensor has 2 completely masked rows, so the last 2 rows of the _
        # sorted_ states will be completely zero, having been appended after
        # calling the respective encoder.
        new_states[0][:, -2:, :] = 0
        new_states[1][:, -2:, :] = 0

        index_selected_new_states = (
            new_states[0].index_select(1, self.restoration_indices),
            new_states[1].index_select(1, self.restoration_indices),
        )

        self.encoder_base._update_states(new_states, self.restoration_indices)
        # Check that the update _preserved_ the state for the rows which were
        # completely masked (2 and 4):
        for index in [2, 4]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                index_selected_initial_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                index_selected_initial_states[1][:, index, :].data.numpy(),
            )
        # Now the states which were updated:
        for index in [0, 1, 3]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                index_selected_new_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                index_selected_new_states[1][:, index, :].data.numpy(),
            )

        # Now test the case that the new state is smaller:
        small_new_states = torch.randn([1, 3, 7]), torch.randn([1, 3, 7])
        # pretend the 2nd sequence in the batch was fully masked.
        small_restoration_indices = torch.LongTensor([2, 0, 1])
        small_new_states[0][:, 0, :] = 0
        small_new_states[1][:, 0, :] = 0

        index_selected_small_states = (
            small_new_states[0].index_select(1, small_restoration_indices),
            small_new_states[1].index_select(1, small_restoration_indices),
        )
        self.encoder_base._update_states(small_new_states,
                                         small_restoration_indices)

        # Check the index for the row we didn't update is the same as the
        # previous step:
        for index in [1, 3]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                index_selected_new_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                index_selected_new_states[1][:, index, :].data.numpy(),
            )
        # Indices we did update:
        for index in [0, 2]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                index_selected_small_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                index_selected_small_states[1][:, index, :].data.numpy(),
            )

        # We didn't update index 4 in the previous step either, so it should
        # be equal to the 4th index of initial states.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0][:, 4, :].data.numpy(),
            index_selected_initial_states[0][:, 4, :].data.numpy(),
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1][:, 4, :].data.numpy(),
            index_selected_initial_states[1][:, 4, :].data.numpy(),
        )

    def test_reset_states(self):
        # Initialize the encoder states.
        assert self.encoder_base._states is None
        initial_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])
        index_selected_initial_states = (
            initial_states[0].index_select(1, self.restoration_indices),
            initial_states[1].index_select(1, self.restoration_indices),
        )
        self.encoder_base._update_states(initial_states,
                                         self.restoration_indices)

        # Check that only some of the states are reset when a mask is provided.
        mask = torch.FloatTensor([1, 1, 0, 0, 0])
        self.encoder_base.reset_states(mask)
        # First two states should be zeros
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0][:, :2, :].data.numpy(),
            torch.zeros_like(initial_states[0])[:, :2, :].data.numpy(),
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1][:, :2, :].data.numpy(),
            torch.zeros_like(initial_states[1])[:, :2, :].data.numpy(),
        )
        # Remaining states should be the same
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0][:, 2:, :].data.numpy(),
            index_selected_initial_states[0][:, 2:, :].data.numpy(),
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1][:, 2:, :].data.numpy(),
            index_selected_initial_states[1][:, 2:, :].data.numpy(),
        )

        # Check that error is raised if mask has wrong batch size.
        bad_mask = torch.FloatTensor([1, 1, 0])
        with self.assertRaises(ValueError):
            self.encoder_base.reset_states(bad_mask)

        # Check that states are reset to None if no mask is provided.
        self.encoder_base.reset_states()
        assert self.encoder_base._states is None

    def test_non_contiguous_initial_states_handled(self):
        # Check that the encoder is robust to non-contiguous initial states.

        # Case 1: Encoder is not stateful

        # A transposition will make the tensors non-contiguous, start them off
        # at the wrong shape and transpose them into the right shape.
        encoder_base = _EncoderBase(stateful=False)
        initial_states = (
            torch.randn(5, 6, 7).permute(1, 0, 2),
            torch.randn(5, 6, 7).permute(1, 0, 2),
        )
        assert not initial_states[0].is_contiguous() and \
               not initial_states[1].is_contiguous()
        assert initial_states[0].size() == torch.Size([6, 5, 7])
        assert initial_states[1].size() == torch.Size([6, 5, 7])

        # We'll pass them through an LSTM encoder and a vanilla RNN encoder to
        # make sure it works whether the initial states are a tuple of tensors
        # or just a single tensor.
        encoder_base.sort_and_run_forward(self.lstm, self.tensor,
                                          self.mask, initial_states)
        encoder_base.sort_and_run_forward(self.rnn, self.tensor,
                                          self.mask, initial_states[0])

        # Case 2: Encoder is stateful

        # For stateful encoders, the initial state may be non-contiguous if
        # its state was previously updated with non-contiguous tensors. As in
        # the non-stateful tests, we check that the encoder still works on
        # initial states for RNNs and LSTMs.
        final_states = initial_states
        # Check LSTM
        encoder_base = _EncoderBase(stateful=True)
        encoder_base._update_states(final_states, self.restoration_indices)
        encoder_base.sort_and_run_forward(self.lstm, self.tensor, self.mask)
        # Check RNN
        encoder_base.reset_states()
        encoder_base._update_states([final_states[0]], self.restoration_indices)
        encoder_base.sort_and_run_forward(self.rnn, self.tensor, self.mask)

    @cuda_test
    def test_non_contiguous_initial_states_handled_on_gpu(self):
        # Some PyTorch operations which produce contiguous tensors on the CPU
        # produce non-contiguous tensors on the GPU (e.g. forward pass of an
        # RNN when batch_first=True). Accordingly, we perform the same checks
        # from previous test on the GPU to ensure the encoder is not affected
        # by which device it is on.

        # Case 1: Encoder is not stateful

        # A transposition will make the tensors non-contiguous, start them off
        # at the wrong shape and transpose them into the right shape.
        encoder_base = _EncoderBase(stateful=False).cuda()
        initial_states = (
            torch.randn(5, 6, 7).cuda().permute(1, 0, 2),
            torch.randn(5, 6, 7).cuda().permute(1, 0, 2),
        )
        assert not initial_states[0].is_contiguous() and not initial_states[
            1].is_contiguous()
        assert initial_states[0].size() == torch.Size([6, 5, 7])
        assert initial_states[1].size() == torch.Size([6, 5, 7])

        # We'll pass them through an LSTM encoder and a vanilla RNN encoder to
        # make sure it works whether the initial states are a tuple of tensors
        # or just a single tensor.
        encoder_base.sort_and_run_forward(
            self.lstm.cuda(), self.tensor.cuda(), self.mask.cuda(),
            initial_states
        )
        encoder_base.sort_and_run_forward(
            self.rnn.cuda(), self.tensor.cuda(), self.mask.cuda(),
            initial_states[0]
        )

        # Case 2: Encoder is stateful

        # For stateful encoders, the initial state may be non-contiguous if its
        # state was previously updated with non-contiguous tensors. As in the
        # non-stateful tests, we check that the encoder still works on initial
        # states for RNNs and LSTMs.
        final_states = initial_states
        # Check LSTM
        encoder_base = _EncoderBase(stateful=True).cuda()
        encoder_base._update_states(final_states,
                                    self.restoration_indices.cuda())
        encoder_base.sort_and_run_forward(self.lstm.cuda(), self.tensor.cuda(),
                                          self.mask.cuda())
        # Check RNN
        encoder_base.reset_states()
        encoder_base._update_states([final_states[0]],
                                    self.restoration_indices.cuda())
        encoder_base.sort_and_run_forward(self.rnn.cuda(), self.tensor.cuda(),
                                          self.mask.cuda())


class TestHighway(unittest.TestCase):

    def test_forward_works_on_simple_input(self):
        highway = Highway(2, 2)

        highway._layers[0].weight.data.fill_(1)
        highway._layers[0].bias.data.fill_(0)
        highway._layers[1].weight.data.fill_(2)
        highway._layers[1].bias.data.fill_(-2)
        input_tensor = torch.FloatTensor([[-2, 1], [3, -2]])
        result = highway(input_tensor).data.numpy()
        assert result.shape == (2, 2)
        # This was checked by hand.
        assert_almost_equal(result, [[-0.0394, 0.0197], [1.7527, -0.5550]],
                            decimal=4)

    def test_forward_works_on_nd_input(self):
        highway = Highway(2, 2)
        input_tensor = torch.ones(2, 2, 2)
        output = highway(input_tensor)
        assert output.size() == (2, 2, 2)


class TestLstmCellWithProjection(unittest.TestCase):

    def test_elmo_lstm_cell_completes_forward_pass(self):
        input_tensor = torch.rand(4, 5, 3)
        input_tensor[1, 4:, :] = 0.0
        input_tensor[2, 2:, :] = 0.0
        input_tensor[3, 1:, :] = 0.0

        initial_hidden_state = torch.ones([1, 4, 5])
        initial_memory_state = torch.ones([1, 4, 7])

        lstm = LstmCellWithProjection(
            input_size=3,
            hidden_size=5,
            cell_size=7,
            memory_cell_clip_value=2,
            state_projection_clip_value=1,
        )
        output_sequence, lstm_state = lstm(
            input_tensor, [5, 4, 2, 1], (initial_hidden_state,
                                         initial_memory_state)
        )
        numpy.testing.assert_array_equal(
            output_sequence.data[1, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(
            output_sequence.data[2, 2:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(
            output_sequence.data[3, 1:, :].numpy(), 0.0)

        # Test the state clipping.
        numpy.testing.assert_array_less(output_sequence.data.numpy(), 1.0)
        numpy.testing.assert_array_less(-output_sequence.data.numpy(), 1.0)

        # LSTM state should be (num_layers, batch_size, hidden_size)
        assert list(lstm_state[0].size()) == [1, 4, 5]
        # LSTM memory cell should be (num_layers, batch_size, cell_size)
        assert list((lstm_state[1].size())) == [1, 4, 7]

        # Test the cell clipping.
        numpy.testing.assert_array_less(lstm_state[0].data.numpy(), 2.0)
        numpy.testing.assert_array_less(-lstm_state[0].data.numpy(), 2.0)


class TestTimeDistributed(unittest.TestCase):

    def test_time_distributed_reshapes_named_arg_correctly(self):
        char_embedding = Embedding(2, 2)
        char_embedding.weight = Parameter(
            torch.FloatTensor([[0.4, 0.4], [0.5, 0.5]]))
        distributed_embedding = TimeDistributed(char_embedding)
        char_input = torch.LongTensor([[[1, 0], [1, 1]]])
        output = distributed_embedding(char_input)
        assert_almost_equal(
            output.data.numpy(),
            [[[[0.5, 0.5], [0.4, 0.4]], [[0.5, 0.5], [0.5, 0.5]]]]
        )

    def test_time_distributed_reshapes_positional_kwarg_correctly(self):
        char_embedding = Embedding(2, 2)
        char_embedding.weight = Parameter(torch.FloatTensor(
            [[0.4, 0.4], [0.5, 0.5]]))
        distributed_embedding = TimeDistributed(char_embedding)
        char_input = torch.LongTensor([[[1, 0], [1, 1]]])
        output = distributed_embedding(input=char_input)
        assert_almost_equal(
            output.data.numpy(),
            [[[[0.5, 0.5], [0.4, 0.4]], [[0.5, 0.5], [0.5, 0.5]]]]
        )

    def test_time_distributed_works_with_multiple_inputs(self):
        module = lambda x, y: x + y
        distributed = TimeDistributed(module)
        x_input = torch.LongTensor([[[1, 2], [3, 4]]])
        y_input = torch.LongTensor([[[4, 2], [9, 1]]])
        output = distributed(x_input, y_input)
        assert_almost_equal(output.data.numpy(), [[[5, 4], [12, 5]]])

    def test_time_distributed_reshapes_multiple_inputs_with_pass_through_tensor_correctly(self):

        class FakeModule(Module):

            def forward(self, input_tensor, tensor_to_pass_through=None,
                        another_tensor=None):

                return input_tensor + tensor_to_pass_through + another_tensor

        module = FakeModule()
        distributed_module = TimeDistributed(module)

        input_tensor1 = torch.LongTensor([[[1, 2], [3, 4]]])
        input_to_pass_through = torch.LongTensor([3, 7])
        input_tensor2 = torch.LongTensor([[[4, 2], [9, 1]]])

        output = distributed_module(
            input_tensor1,
            tensor_to_pass_through=input_to_pass_through,
            another_tensor=input_tensor2,
            pass_through=["tensor_to_pass_through"],
        )
        assert_almost_equal(output.data.numpy(), [[[8, 11], [15, 12]]])

    def test_time_distributed_reshapes_multiple_inputs_with_pass_through_non_tensor_correctly(self):

        class FakeModule(Module):

            def forward(self, input_tensor, number=0, another_tensor=None):

                return input_tensor + number + another_tensor

        module = FakeModule()
        distributed_module = TimeDistributed(module)

        input_tensor1 = torch.LongTensor([[[1, 2], [3, 4]]])
        input_number = 5
        input_tensor2 = torch.LongTensor([[[4, 2], [9, 1]]])

        output = distributed_module(
            input_tensor1,
            number=input_number,
            another_tensor=input_tensor2,
            pass_through=["number"],
        )
        assert_almost_equal(output.data.numpy(), [[[10, 9], [17, 10]]])


class TestUtils(unittest.TestCase):

    def test_add_sentence_boundary_token_ids_handles_2D_input(self):
        tensor = torch.from_numpy(numpy.array([[1, 2, 3], [4, 5, 0]]))
        mask = (tensor > 0).long()
        bos = 9
        eos = 10
        new_tensor, new_mask = add_sentence_boundary_token_ids(
            tensor, mask, bos, eos)
        expected_new_tensor = numpy.array([[9, 1, 2, 3, 10], [9, 4, 5, 10, 0]])
        assert (new_tensor.data.numpy() == expected_new_tensor).all()
        assert (new_mask.data.numpy() == (expected_new_tensor > 0)).all()

    def test_add_sentence_boundary_token_ids_handles_3D_input(self):
        tensor = torch.from_numpy(
            numpy.array(
                [
                    [[1, 2, 3, 4], [5, 5, 5, 5], [6, 8, 1, 2]],
                    [[4, 3, 2, 1], [8, 7, 6, 5], [0, 0, 0, 0]],
                ]
            )
        )
        mask = ((tensor > 0).sum(dim=-1) > 0).type(torch.LongTensor)
        bos = torch.from_numpy(numpy.array([9, 9, 9, 9]))
        eos = torch.from_numpy(numpy.array([10, 10, 10, 10]))
        new_tensor, new_mask = add_sentence_boundary_token_ids(
            tensor, mask, bos, eos)
        expected_new_tensor = numpy.array(
            [
                [[9, 9, 9, 9], [1, 2, 3, 4], [5, 5, 5, 5], [6, 8, 1, 2],
                 [10, 10, 10, 10]],
                [[9, 9, 9, 9], [4, 3, 2, 1], [8, 7, 6, 5], [10, 10, 10, 10],
                 [0, 0, 0, 0]],
            ]
        )
        assert (new_tensor.data.numpy() == expected_new_tensor).all()
        assert (new_mask.data.numpy() == (
                (expected_new_tensor > 0).sum(axis=-1) > 0)).all()

    def test_remove_sentence_boundaries(self):
        tensor = torch.from_numpy(numpy.random.rand(3, 5, 7))
        mask = torch.from_numpy(
            # The mask with two elements is to test the corner case
            # of an empty sequence, so here we are removing boundaries
            # from  "<S> </S>"
            numpy.array([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        ).long()
        new_tensor, new_mask = remove_sentence_boundaries(tensor, mask)

        expected_new_tensor = torch.zeros(3, 3, 7)
        expected_new_tensor[1, 0:3, :] = tensor[1, 1:4, :]
        expected_new_tensor[2, 0:2, :] = tensor[2, 1:3, :]
        assert_array_almost_equal(new_tensor.data.numpy(),
                                  expected_new_tensor.data.numpy())

        expected_new_mask = torch.from_numpy(numpy.array(
            [[0, 0, 0], [1, 1, 1], [1, 1, 0]])).long()
        assert (new_mask.data.numpy() == expected_new_mask.data.numpy()).all()

    def test_block_orthogonal_can_initialize(self):
        tensor = torch.zeros([10, 6])
        block_orthogonal(tensor, [5, 3])
        tensor = tensor.data.numpy()

        def test_block_is_orthogonal(block) -> None:
            matrix_product = block.T @ block
            numpy.testing.assert_array_almost_equal(
                matrix_product, numpy.eye(matrix_product.shape[-1]), 6
            )

        test_block_is_orthogonal(tensor[:5, :3])
        test_block_is_orthogonal(tensor[:5, 3:])
        test_block_is_orthogonal(tensor[5:, 3:])
        test_block_is_orthogonal(tensor[5:, :3])

    def test_block_orthogonal_raises_on_mismatching_dimensions(self):
        tensor = torch.zeros([10, 6, 8])
        with self.assertRaises(ConfigurationError):
            block_orthogonal(tensor, [7, 2, 1])

    def test_combine_initial_dims(self):
        tensor = torch.randn(4, 10, 20, 17, 5)

        tensor2d = combine_initial_dims(tensor)
        assert list(tensor2d.size()) == [4 * 10 * 20 * 17, 5]

    def test_uncombine_initial_dims(self):
        embedding2d = torch.randn(4 * 10 * 20 * 17 * 5, 12)

        embedding = uncombine_initial_dims(embedding2d,
                                           torch.Size((4, 10, 20, 17, 5)))
        assert list(embedding.size()) == [4, 10, 20, 17, 5, 12]


class TestScalarMix(unittest.TestCase):

    def test_scalar_mix_can_run_forward(self):
        mixture = ScalarMix(3)
        tensors = [torch.randn([3, 4, 5]) for _ in range(3)]
        for k in range(3):
            mixture.scalar_parameters[k].data[0] = 0.1 * (k + 1)
        mixture.gamma.data[0] = 0.5
        result = mixture(tensors)

        weights = [0.1, 0.2, 0.3]
        normed_weights = numpy.exp(weights) / numpy.sum(numpy.exp(weights))
        expected_result = sum(normed_weights[k] * tensors[k].data.numpy()
                              for k in range(3))
        expected_result *= 0.5
        numpy.testing.assert_almost_equal(expected_result, result.data.numpy())

    def test_scalar_mix_throws_error_on_incorrect_number_of_inputs(self):
        mixture = ScalarMix(3)
        tensors = [torch.randn([3, 4, 5]) for _ in range(5)]
        with self.assertRaises(ConfigurationError):
            _ = mixture(tensors)

    def test_scalar_mix_throws_error_on_incorrect_initial_scalar_parameters_length(self):
        with self.assertRaises(ConfigurationError):
            ScalarMix(3, initial_scalar_parameters=[0.0, 0.0])

    def test_scalar_mix_trainable_with_initial_scalar_parameters(self):
        initial_scalar_parameters = [1.0, 2.0, 3.0]
        mixture = ScalarMix(3,
                            initial_scalar_parameters=initial_scalar_parameters,
                            trainable=False)
        for i, scalar_mix_parameter in enumerate(mixture.scalar_parameters):
            assert scalar_mix_parameter.requires_grad is False
            assert scalar_mix_parameter.item() == initial_scalar_parameters[i]

    def test_scalar_mix_layer_norm(self):
        mixture = ScalarMix(3, do_layer_norm="scalar_norm_reg")

        tensors = [torch.randn([3, 4, 5]) for _ in range(3)]
        numpy_mask = numpy.ones((3, 4), dtype="int32")
        numpy_mask[1, 2:] = 0
        mask = torch.from_numpy(numpy_mask)

        weights = [0.1, 0.2, 0.3]
        for k in range(3):
            mixture.scalar_parameters[k].data[0] = weights[k]
        mixture.gamma.data[0] = 0.5
        result = mixture(tensors, mask)

        normed_weights = numpy.exp(weights) / numpy.sum(numpy.exp(weights))
        expected_result = numpy.zeros((3, 4, 5))
        for k in range(3):
            mean = numpy.mean(tensors[k].data.numpy()[numpy_mask == 1])
            std = numpy.std(tensors[k].data.numpy()[numpy_mask == 1])
            normed_tensor = (tensors[k].data.numpy() - mean) / (std + 1e-12)
            expected_result += normed_tensor * normed_weights[k]
        expected_result *= 0.5

        numpy.testing.assert_almost_equal(expected_result, result.data.numpy(),
                                          decimal=6)


if __name__ == "__main__":
    unittest.main()
