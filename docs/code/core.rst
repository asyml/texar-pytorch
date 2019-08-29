.. role:: hidden
    :class: hidden-section

Core
****

.. _attention-mechanism:

Attention Mechanism
===================

:hidden:`AttentionWrapperState`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.AttentionWrapperState
    :members:

:hidden:`LuongAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.LuongAttention
    :members:

:hidden:`BahdanauAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.BahdanauAttention
    :members:

:hidden:`BahdanauMonotonicAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.BahdanauMonotonicAttention
    :members:

:hidden:`LuongMonotonicAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.LuongMonotonicAttention
    :members:

:hidden:`compute_attention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.compute_attention

:hidden:`monotonic_attention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.monotonic_attention

:hidden:`hardmax`
~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.hardmax

:hidden:`sparsemax`
~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.sparsemax



Cells
=====

:hidden:`default_rnn_cell_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.default_rnn_cell_hparams

:hidden:`get_rnn_cell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_rnn_cell

:hidden:`wrap_builtin_cell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.wrap_builtin_cell

:hidden:`RNNCellBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.cell_wrappers.RNNCellBase
    :members:

:hidden:`RNNCell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.cell_wrappers.RNNCell
    :members:

:hidden:`GRUCell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.cell_wrappers.GRUCell
    :members:

:hidden:`LSTMCell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.cell_wrappers.LSTMCell
    :members:

:hidden:`DropoutWrapper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.cell_wrappers.DropoutWrapper
    :members:

:hidden:`ResidualWrapper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.cell_wrappers.ResidualWrapper
    :members:

:hidden:`HighwayWrapper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.cell_wrappers.HighwayWrapper
    :members:

:hidden:`MultiRNNCell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.cell_wrappers.MultiRNNCell
    :members:

:hidden:`AttentionWrapper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. spelling::
    Luong
    Bahdanau

.. autoclass:: texar.torch.core.cell_wrappers.AttentionWrapper
    :members:



Layers
======

:hidden:`get_layer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_layer

:hidden:`MaxReducePool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.MaxReducePool1d
    :members:

:hidden:`AvgReducePool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.AvgReducePool1d
    :members:

:hidden:`get_pooling_layer_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_pooling_layer_hparams

:hidden:`MergeLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.MergeLayer
    :members:

:hidden:`Flatten`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.Flatten
    :members:
    :exclude-members: forward

:hidden:`Identity`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.core.Identity
    :members:
    :exclude-members: forward

:hidden:`default_regularizer_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.default_regularizer_hparams

:hidden:`get_regularizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_regularizer

:hidden:`get_initializer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_initializer

:hidden:`get_activation_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_activation_fn


Optimization
=============

:hidden:`default_optimization_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.default_optimization_hparams

:hidden:`get_train_op`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_train_op

:hidden:`get_scheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_scheduler

:hidden:`get_optimizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_optimizer

:hidden:`get_grad_clip_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.core.get_grad_clip_fn
