"""
Unit tests for connectors.
"""

from __future__ import unicode_literals

from collections import namedtuple
import unittest
import torch
import torch.distributions as tds

from texar.torch.core import layers
from texar.torch.modules.connectors.connectors import ConstantConnector
from texar.torch.modules.connectors.connectors import MLPTransformConnector
from texar.torch.modules.connectors.connectors import (
    ReparameterizedStochasticConnector)
from texar.torch.modules.connectors.connectors import _assert_same_size

from texar.torch.utils import nest


class TestConnectors(unittest.TestCase):
    r"""Tests various connectors.
    """

    def setUp(self) -> None:
        self._batch_size = 100

        self._decoder_cell = layers.get_rnn_cell(
            256, layers.default_rnn_cell_hparams())

    def test_constant_connector(self):
        r"""Tests the logic of
        :class:`~texar.torch.modules.connectors.ConstantConnector`.
        """

        state_size = namedtuple('LSTMStateTuple', ['h', 'c'])(256, 256)
        connector_0 = ConstantConnector(state_size)
        decoder_initial_state_0 = connector_0(self._batch_size)
        connector_1 = ConstantConnector(
            state_size, hparams={"value": 1.})
        decoder_initial_state_1 = connector_1(self._batch_size)

        s_0 = decoder_initial_state_0
        s_1 = decoder_initial_state_1
        self.assertEqual(nest.flatten(s_0)[0][0, 0], 0.)
        self.assertEqual(nest.flatten(s_1)[0][0, 0], 1.)

        size = torch.Size([1, 2, 3])
        connector_size_0 = ConstantConnector(
            size, hparams={"value": 2.})
        size_tensor = connector_size_0(self._batch_size)
        self.assertEqual(
            torch.Size([self._batch_size]) + size, size_tensor.size())
        self.assertEqual(size_tensor[0][0, 0, 0], 2.)

        tuple_size_1 = (torch.Size([1, 2, 3]), torch.Size([4, 5, 6]))
        connector_size_1 = ConstantConnector(
            tuple_size_1, hparams={"value": 3.})
        tuple_size_tensor = connector_size_1(self._batch_size)
        tuple_size_tensor_0 = tuple_size_tensor[0]
        tuple_size_tensor_1 = tuple_size_tensor[1]
        self.assertEqual(
            torch.Size([self._batch_size]) + torch.Size([1, 2, 3]),
            tuple_size_tensor_0.size())
        self.assertEqual(tuple_size_tensor_0[0][0, 0, 0], 3.)
        self.assertEqual(
            torch.Size([self._batch_size]) + torch.Size([4, 5, 6]),
            tuple_size_tensor_1.size())
        self.assertEqual(tuple_size_tensor_1[0][0, 0, 0], 3.)

        tuple_size_2 = (5, 10)
        connector_size_2 = ConstantConnector(
            tuple_size_2, hparams={"value": 4.})
        tuple_size_tensor = connector_size_2(self._batch_size)
        tuple_size_tensor_0 = tuple_size_tensor[0]
        tuple_size_tensor_1 = tuple_size_tensor[1]
        self.assertEqual(
            torch.Size([self._batch_size]) + torch.Size([5]),
            tuple_size_tensor_0.size())
        self.assertEqual(tuple_size_tensor_0[0][0], 4.)
        self.assertEqual(
            torch.Size([self._batch_size]) + torch.Size([10]),
            tuple_size_tensor_1.size())
        self.assertEqual(tuple_size_tensor_1[0][0], 4.)

        tuple_size_3 = (torch.Size([1, 2, 3]), 10)
        connector_size_3 = ConstantConnector(
            tuple_size_3, hparams={"value": 4.})
        tuple_size_tensor = connector_size_3(self._batch_size)
        tuple_size_tensor_0 = tuple_size_tensor[0]
        tuple_size_tensor_1 = tuple_size_tensor[1]
        self.assertEqual(
            torch.Size([self._batch_size]) + torch.Size([1, 2, 3]),
            tuple_size_tensor_0.size())
        self.assertEqual(tuple_size_tensor_0[0][0, 0, 0], 4.)
        self.assertEqual(
            torch.Size([self._batch_size]) + torch.Size([10]),
            tuple_size_tensor_1.size())
        self.assertEqual(tuple_size_tensor_1[0][0], 4.)

    def test_forward_connector(self):
        r"""Tests the logic of
        :class:`~texar.torch.modules.connectors.ForwardConnector`.
        """
        # TODO(zhiting)
        pass

    def test_mlp_transform_connector(self):
        r"""Tests the logic of
        :class:`~texar.torch.modules.connectors.MLPTransformConnector`.
        """
        state_size = namedtuple('LSTMStateTuple', ['h', 'c'])(256, 256)
        connector = MLPTransformConnector(state_size, linear_layer_dim=10)
        output = connector(torch.zeros(5, 10))
        output_1 = output[0]
        output_2 = output[1]
        self.assertEqual(
            torch.Size([5, 256]),
            output_1.size())
        self.assertEqual(
            torch.Size([5, 256]),
            output_2.size())

        state_size = (16, 32)
        connector = MLPTransformConnector(state_size, linear_layer_dim=10)
        output = connector(torch.zeros(5, 10))
        output_1 = output[0]
        output_2 = output[1]
        self.assertEqual(
            torch.Size([5, 16]),
            output_1.size())
        self.assertEqual(
            torch.Size([5, 32]),
            output_2.size())

        state_size = (torch.Size([8, 32]), torch.Size([16, 64]))
        connector = MLPTransformConnector(state_size, linear_layer_dim=10)
        output = connector(torch.zeros(5, 10))
        output_1 = output[0]
        output_2 = output[1]
        self.assertEqual(
            torch.Size([5, 8, 32]),
            output_1.size())
        self.assertEqual(
            torch.Size([5, 16, 64]),
            output_2.size())

    def test_reparameterized_stochastic_connector(self):
        r"""Tests the logic of
        :class:`~texar.torch.modules.ReparameterizedStochasticConnector`.
        """

        self._batch_size = 16
        variable_size = 100
        sample_num = 10

        mu = torch.zeros([self._batch_size, variable_size])
        var = torch.ones([variable_size])

        state_size = (10, 11)
        gauss_connector = ReparameterizedStochasticConnector(
            state_size,
            mlp_input_size=mu.size()[-1:],
            distribution="MultivariateNormal",
            distribution_kwargs={"loc": mu, "scale_tril": torch.diag(var)})
        output, samples = gauss_connector()

        self.assertEqual(output[0].size(),
                         torch.Size([self._batch_size, state_size[0]]))
        self.assertEqual(output[1].size(),
                         torch.Size([self._batch_size, state_size[1]]))
        self.assertEqual(samples.size(),
                         torch.Size([self._batch_size, variable_size]))

        output, _ = gauss_connector(num_samples=12, transform=False)
        self.assertEqual(output.size(),
                         torch.Size([12, self._batch_size, variable_size]))

        state_size_ts = (torch.Size([10, 10]), torch.Size([2, 3, 4]))
        gauss_connector_ts = ReparameterizedStochasticConnector(
            state_size_ts,
            mlp_input_size=mu.size()[-1:],
            distribution="MultivariateNormal",
            distribution_kwargs={"loc": mu, "scale_tril": torch.diag(var)})
        output, samples = gauss_connector_ts()

        _assert_same_size(output, state_size_ts)

        # sample_mu = np.mean(sample_outputs, axis=0)
        # # pylint: disable=no-member
        # sample_var = np.var(sample_outputs, axis=0)

        # # check if the value is approximated N(0, 1)
        # for i in range(variable_size):
        #     self.assertAlmostEqual(0, sample_mu[i], delta=0.2)
        #     self.assertAlmostEqual(1, sample_var[i], delta=0.2)


if __name__ == "__main__":
    unittest.main()
