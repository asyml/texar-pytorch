.. role:: hidden
    :class: hidden-section

Core
****


Attention Mechanism
===================

:hidden:`AttentionWrapperState`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.AttentionWrapperState
    :members:

:hidden:`LuongAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.LuongAttention
    :members:

:hidden:`BahdanauAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.BahdanauAttention
    :members:

:hidden:`BahdanauMonotonicAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.BahdanauMonotonicAttention
    :members:

:hidden:`LuongMonotonicAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.LuongMonotonicAttention
    :members:

:hidden:`compute_attention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.compute_attention

:hidden:`monotonic_attention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.monotonic_attention

:hidden:`hardmax`
~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.hardmax

:hidden:`sparsemax`
~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.sparsemax



Cells
=====

:hidden:`default_rnn_cell_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.default_rnn_cell_hparams 

:hidden:`get_rnn_cell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_rnn_cell

:hidden:`wrap_builtin_cell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.wrap_builtin_cell

:hidden:`RNNCellBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.RNNCellBase
    :members:

:hidden:`RNNCell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.RNNCell
    :members:

:hidden:`GRUCell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.GRUCell
    :members:

:hidden:`LSTMCell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.LSTMCell
    :members:

:hidden:`DropoutWrapper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.DropoutWrapper
    :members:

:hidden:`ResidualWrapper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.ResidualWrapper
    :members:

:hidden:`HighwayWrapper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.HighwayWrapper
    :members:

:hidden:`MultiRNNCell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.MultiRNNCell
    :members:

:hidden:`AttentionWrapper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.cell_wrappers.AttentionWrapper
    :members:



Layers
======

:hidden:`get_layer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_layer

:hidden:`MaxReducePool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.MaxReducePool1d
    :members:

:hidden:`AvgReducePool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.AvgReducePool1d
    :members:

:hidden:`get_pooling_layer_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_pooling_layer_hparams

:hidden:`MergeLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.MergeLayer
    :members:

:hidden:`Flatten`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.Flatten
    :members:
    :exclude-members: forward

:hidden:`Identity`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.Identity
    :members:
    :exclude-members: forward

:hidden:`default_regularizer_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.default_regularizer_hparams

:hidden:`get_regularizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_regularizer

:hidden:`get_initializer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_initializer

:hidden:`get_activation_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_activation_fn


Optimization
=============

:hidden:`default_optimization_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.default_optimization_hparams

:hidden:`get_train_op`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_train_op

:hidden:`get_scheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_scheduler

:hidden:`get_optimizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_optimizer

:hidden:`get_grad_clip_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_grad_clip_fn
