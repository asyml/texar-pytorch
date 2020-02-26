.. role:: hidden
    :class: hidden-section

Executor
***********

Executor
==========

:hidden:`Executor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.Executor
    :members:
    :private-members:


Conditions
=============

:hidden:`Event`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.condition.Event

:hidden:`Condition`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.condition.Condition

:hidden:`epoch`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.condition.epoch

:hidden:`iteration`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.condition.iteration

:hidden:`validation`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.condition.validation

:hidden:`consecutive`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.condition.consecutive

:hidden:`once`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.condition.once

:hidden:`time`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.condition.time


Metrics
========

:hidden:`Metric`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.Metric
    :members:

:hidden:`SimpleMetric`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.SimpleMetric
    :members:

:hidden:`StreamingMetric`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.StreamingMetric
    :members:

:hidden:`Accuracy`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.Accuracy

:hidden:`ConfusionMatrix`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.ConfusionMatrix
    :members: class_id

:hidden:`Precision`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.Precision

:hidden:`Recall`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.Recall

:hidden:`F1`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.F1

:hidden:`PearsonR`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.PearsonR

:hidden:`RMSE`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.RMSE

:hidden:`Average`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.Average

:hidden:`AveragePerplexity`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.AveragePerplexity

:hidden:`RunningAverage`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.RunningAverage

:hidden:`LR`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.metric.LR


Actions
========

:hidden:`reset_params`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.action.reset_params

:hidden:`scale_lr`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.action.scale_lr

:hidden:`early_stop`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.run.action.early_stop
