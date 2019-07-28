"""
Unit tests for various optimization related utilities.
"""

import unittest

import torch

from texar.torch.core.optimization import *


class OptimizationTest(unittest.TestCase):
    r"""Test optimization.
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
        r"""Tests get_optimizer.
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

    def test_get_scheduler(self):
        r"""Tests get_scheduler.
        """
        optimizer = get_optimizer(params=[torch.tensor(1)], hparams=None)

        default_scheduler = get_scheduler(optimizer=optimizer,
                                          hparams=None)
        self.assertEqual(default_scheduler, None)

        hparams = {
            "optimizer": {
                "type": "",
                "kwargs": {}
            },
            "learning_rate_decay": {
                "type": "ExponentialLR",
                "kwargs": {
                    "gamma": 0.99
                }
            },
            "gradient_clip": {
                "type": "",
                "kwargs": {}
            },
            "gradient_noise_scale": None,
            "name": None
        }

        scheduler = get_scheduler(optimizer=optimizer,
                                  hparams=hparams)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)

        hparams = {
            "optimizer": {
                "type": "",
                "kwargs": {}
            },
            "learning_rate_decay": {
                "type": torch.optim.lr_scheduler.ExponentialLR,
                "kwargs": {
                    "gamma": 0.99
                }
            },
            "gradient_clip": {
                "type": "",
                "kwargs": {}
            },
            "gradient_noise_scale": None,
            "name": None
        }

        scheduler = get_scheduler(optimizer=optimizer,
                                  hparams=hparams)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_get_grad_clip_fn(self):
        r"""Tests get_grad_clip_fn.
        """
        default_grad_clip_fn = get_grad_clip_fn(hparams=None)
        self.assertEqual(default_grad_clip_fn, None)

        hparams = {
            "optimizer": {
                "type": "",
                "kwargs": {}
            },
            "learning_rate_decay": {
                "type": "",
                "kwargs": {}
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

        grad_clip_fn = get_grad_clip_fn(hparams=hparams)
        if not callable(grad_clip_fn):
            raise ValueError("grad_clip_fn is not callable")

        hparams = {
            "optimizer": {
                "type": "",
                "kwargs": {}
            },
            "learning_rate_decay": {
                "type": "",
                "kwargs": {}
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

        grad_clip_fn = get_grad_clip_fn(hparams=hparams)
        if not callable(grad_clip_fn):
            raise ValueError("grad_clip_fn is not callable")

    def test_get_train_op(self):
        r"""Tests get_train_op.
        """
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

        # Case 1
        optimizer = get_optimizer(self.model.parameters(), hparams)
        train_op = get_train_op(optimizer=optimizer, hparams=hparams)

        for t in range(50):
            y_pred = self.model(self.x)
            loss = self.loss_fn(y_pred, self.y)
            loss.backward()
            train_op()

        # Case 2
        train_op = get_train_op(params=self.model.parameters(), hparams=hparams)

        for t in range(50):
            y_pred = self.model(self.x)
            loss = self.loss_fn(y_pred, self.y)
            loss.backward()
            train_op()

        # Case 3
        optimizer = get_optimizer(self.model.parameters(), hparams)
        scheduler = get_scheduler(optimizer=optimizer,
                                  hparams=hparams)
        train_op = get_train_op(scheduler=scheduler, hparams=hparams)

        for t in range(50):
            y_pred = self.model(self.x)
            loss = self.loss_fn(y_pred, self.y)
            loss.backward()
            train_op()

    def test_BertAdam(self):
        r"""Tests BertAdam.
        """
        optimizer = BertAdam(self.model.parameters())

        for t in range(50):
            y_pred = self.model(self.x)
            loss = self.loss_fn(y_pred, self.y)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
