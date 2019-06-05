"""
Unit tests for various optimization related utilities.
"""

import unittest

import torch

from texar.core.optimization import *


class OptimizationTest(unittest.TestCase):
    """Test optimization.
    """

    def setUp(self):
        N, D_in, H, D_out = 64, 100, 10, 1

        self.x = torch.randn(N, D_in)
        self.y = torch.randn(N, D_out)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')


    def test_get_optimizer(self):
        """Tests get_optimizer.
        """
        default_optimizer = get_optimizer(params=[torch.tensor(1)],
                                          hparams=None)
        self.assertIsInstance(default_optimizer, torch.optim.Adam)

        hparams = {
            "optimizer": {
                "type": "RMSprop",
                "kwargs": {
                    "lr": 0.001,
                    "alpha": 0.99,
                    "eps": 1e-8,
                    "weight_decay": 0,
                    "momentum": 0,
                    "centered": False
                }
            },
            "learning_rate_decay": {
                "type": "",
                "kwargs": {}
            },
            "gradient_clip": {
                "type": "",
                "kwargs": {}
            },
            "gradient_noise_scale": None,
            "name": None
        }

        rmsprop_optimizer = get_optimizer(params=[torch.tensor(1)],
                                          hparams=hparams)
        self.assertIsInstance(rmsprop_optimizer, torch.optim.RMSprop)

        hparams = {
            "optimizer": {
                "type": torch.optim.SGD,
                "kwargs": {
                    "lr": 0.001,
                    "weight_decay": 0,
                    "momentum": 0
                }
            },
            "learning_rate_decay": {
                "type": "",
                "kwargs": {}
            },
            "gradient_clip": {
                "type": "",
                "kwargs": {}
            },
            "gradient_noise_scale": None,
            "name": None
        }

        sgd_optimizer = get_optimizer(params=[torch.tensor(1)],
                                      hparams=hparams)
        self.assertIsInstance(sgd_optimizer, torch.optim.SGD)

    def _run_loop(self, train_op):

        for t in range(500):
            y_pred = self.model(self.x)
            loss = self.loss_fn(y_pred, self.y)
            train_op(loss)

    def test_get_train_op(self):
        """Tests get_train_op.
        """

        hparams = {
            "optimizer": {
                "type": torch.optim.SGD,
                "kwargs": {
                    "lr": 0.001
                }
            },
            "learning_rate_decay": {
                "type": "ExponentialLR",
                "kwargs": {
                    "gamma": 0.99
                }
            },
            "gradient_clip": {
                "type": "clip_grad_norm_",
                "kwargs": {
                    "max_norm": 10,
                    "norm_type": 2
                }
            },
            "gradient_noise_scale": None,
            "name": None
        }

        optimizer = get_optimizer(self.model.parameters(), hparams)
        train_op = get_train_op(optimizer, hparams)
        self._run_loop(train_op)

        hparams = {
            "optimizer": {
                "type": torch.optim.SGD,
                "kwargs": {
                    "lr": 0.001
                }
            },
            "learning_rate_decay": {
                "type": torch.optim.lr_scheduler.ExponentialLR,
                "kwargs": {
                    "gamma": 0.99
                }
            },
            "gradient_clip": {
                "type": torch.nn.utils.clip_grad_norm_,
                "kwargs": {
                    "max_norm": 10,
                    "norm_type": 2
                }
            },
            "gradient_noise_scale": None,
            "name": None
        }

        optimizer = get_optimizer(self.model.parameters(), hparams)
        train_op = get_train_op(optimizer, hparams)
        self._run_loop(train_op)

        hparams = {
            "optimizer": {
                "type": torch.optim.SGD,
                "kwargs": {
                    "lr": 0.001
                }
            },
            "learning_rate_decay": {
                "type": None,
                "kwargs": {}
            },
            "gradient_clip": {
                "type": "",
                "kwargs": {}
            },
            "gradient_noise_scale": None,
            "name": None
        }

        optimizer = get_optimizer(self.model.parameters(), hparams)
        train_op = get_train_op(optimizer, hparams)
        self._run_loop(train_op)


if __name__ == "__main__":
    unittest.main()
