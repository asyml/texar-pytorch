#
"""
Unit tests for connectors.
"""

from __future__ import unicode_literals

from collections import namedtuple
import unittest
import torch
import torch.distributions as tds

from texar.core import layers
from texar.modules.connectors.connectors import ConstantConnector
from texar.modules.connectors.connectors import MLPTransformConnector
from texar.modules.connectors.connectors import (
    ReparameterizedStochasticConnector)
from texar.modules.connectors.connectors import _assert_same_size

from texar.utils import nest


# pylint: disable=too-many-locals, invalid-name

class TestConnectors(unittest.TestCase):
    """Tests various connectors.
    """
    def __init__(self, *args, **kwargs):
        super(TestConnectors, self).__init__(*args, **kwargs)

        self._batch_size = 100

        self._decoder_cell = layers.get_rnn_cell(
            256, layers.default_rnn_cell_hparams())

    def test_constant_connector(self):
        """Tests the logic of
        :class:`~texar.modules.connectors.ConstantConnector`.
        """

        state_size = namedtuple('LSTMStateTuple', ['c', 'h'])(256, 256)
        connector_0 = ConstantConnector(state_size)
        decoder_initial_state_0 = connector_0(self._batch_size)
        connector_1 = ConstantConnector(state_size, hparams={"value": 1.})
        decoder_initial_state_1 = connector_1(self._batch_size)

        s_0 = decoder_initial_state_0
        s_1 = decoder_initial_state_1
        self.assertEqual(nest.flatten(s_0)[0][0, 0], 0.)
        self.assertEqual(nest.flatten(s_1)[0][0, 0], 1.)

        size = torch.Size([1, 2, 3])
        connector_size_0 = ConstantConnector(size, hparams={"value": 2.})
        size_tensor = connector_size_0(self._batch_size)
        self.assertEqual(
            torch.Size([self._batch_size]) + size, size_tensor.size())
        self.assertEqual(size_tensor[0][0, 0, 0], 2.)

        tuple_size = (torch.Size([1, 2, 3]), torch.Size([4, 5, 6]))
        connector_size_1 = ConstantConnector(tuple_size, hparams={"value": 3.})
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





    # pylint: disable=fixme, unnecessary-pass
    def test_forward_connector(self):
        """Tests the logic of
        :class:`~texar.modules.connectors.ForwardConnector`.
        """
        # TODO(zhiting)
        pass

    # pylint: disable=no-self-use
    def test_mlp_transform_connector(self):
        """Tests the logic of
        :class:`~texar.modules.connectors.MLPTransformConnector`.
        """
        state_size = namedtuple('LSTMStateTuple', ['c', 'h'])(256, 256)
        connector = MLPTransformConnector(state_size)
        output = connector(torch.zeros(5, 10))

    def test_reparameterized_stochastic_connector(self):
        """Tests the logic of
        :class:`~texar.modules.ReparameterizedStochasticConnector`.
        """
        self._batch_size = 16
        state_size = (10, 10)
        variable_size = 100
        state_size_ts = (torch.Size([10, 10]), torch.Size([2, 3, 4]))
        sample_num = 10

        mu = torch.zeros([self._batch_size, variable_size])
        var = torch.ones([self._batch_size, variable_size])
        #var = torch.stack([torch.diag(x) for x in var], 0)
        f_mu = torch.flatten(mu)
        f_var = torch.flatten(var)
        mu_vec = torch.zeros([variable_size])
        var_vec = torch.ones([variable_size])
        gauss_ds = tds.multivariate_normal.MultivariateNormal(
            loc=f_mu,
            scale_tril=torch.diag(f_var))
        gauss_ds_vec = tds.multivariate_normal.MultivariateNormal(
            loc=mu_vec,
            scale_tril=torch.diag(var_vec))

        gauss_connector_1 = ReparameterizedStochasticConnector(
            state_size,
            distribution=gauss_ds)
        output_1, _ = gauss_connector_1(num_samples=self._batch_size)

        gauss_connector_2 = ReparameterizedStochasticConnector(
            state_size,
            distribution="MultivariateNormal",
            distribution_kwargs={"loc": f_mu, "scale_tril": torch.diag(f_var)})
        output_2, _ = gauss_connector_2(num_samples=self._batch_size)

        gauss_connector_num = ReparameterizedStochasticConnector(
            state_size,
            distribution=gauss_ds_vec)
        sample_test_num, _ = gauss_connector_num(num_samples=sample_num)

        gauss_connector_ts = ReparameterizedStochasticConnector(
            state_size_ts,
            distribution=gauss_ds)
        sample_ts, _ = gauss_connector_ts()

        self.assertEqual(output_1[0].shape,
                         torch.Size([self._batch_size, state_size[0]]))

        self.assertEqual(output_2[0].shape,
                         torch.Size([self._batch_size, state_size[0]]))

        self.assertEqual(sample_test_num[0].shape,
                         torch.Size([sample_num, state_size[0]]))

        _assert_same_size(sample_ts, state_size_ts)

        # sample_mu = np.mean(sample_outputs, axis=0)
        # # pylint: disable=no-member
        # sample_var = np.var(sample_outputs, axis=0)

        ## check if the value is approximated N(0, 1)
        # for i in range(variable_size):
            # self.assertAlmostEqual(0, sample_mu[i], delta=0.2)
            # self.assertAlmostEqual(1, sample_var[i], delta=0.2)

    #def test_concat_connector(self): # pylint: disable=too-many-locals
    #    """Tests the logic of
    #    :class:`~texar.modules.connectors.ConcatConnector`.
    #    """
    #    gauss_size = 5
    #    constant_size = 7
    #    variable_size = 13

    #    decoder_size1 = 16
    #    decoder_size2 = (16, 32)

    #    gauss_connector = StochasticConnector(gauss_size)
    #    categorical_connector = StochasticConnector(1)
    #    constant_connector = ConstantConnector(constant_size)
    #    concat_connector1 = ConcatConnector(decoder_size1)
    #    concat_connector2 = ConcatConnector(decoder_size2)

    #    # pylint: disable=invalid-name
    #    mu = tf.zeros([self._batch_size, gauss_size])
    #    var = tf.ones([self._batch_size, gauss_size])
    #    categorical_prob = tf.constant(
    #       [[0.1, 0.2, 0.7] for _ in xrange(self._batch_size)])
    #    categorical_ds = tfds.Categorical(probs = categorical_prob)
    #    gauss_ds = tfds.MultivariateNormalDiag(loc = mu, scale_diag = var)

    #    gauss_state = gauss_connector(gauss_ds)
    #    categorical_state = categorical_connector(categorical_ds)
    #    constant_state = constant_connector(self._batch_size, value=1.)
    #    with tf.Session() as debug_sess:
    #        debug_cater = debug_sess.run(categorical_state)

    #    state1 = concat_connector1(
    #       [gauss_state, categorical_state, constant_state])
    #    state2 = concat_connector2(
    #       [gauss_state, categorical_state, constant_state])

    #    with self.test_session() as sess:
    #        sess.run(tf.global_variables_initializer())
    #        [output1, output2] = sess.run([state1, state2])

    #        # check the same size
    #        self.assertEqual(output1.shape[1], decoder_size1)
    #        self.assertEqual(output2[1].shape[1], decoder_size2[1])

if __name__ == "__main__":
    unittest.main()