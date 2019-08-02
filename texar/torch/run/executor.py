import copy
import functools
import pickle
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional, Set, Union, overload

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer

from texar.torch.data.data.data_base import DataBase
from texar.torch.data.data.data_iterators import BatchingStrategy, DataIterator
from texar.torch.data.data.dataset_utils import Batch
from texar.torch.run import executor_utils as utils
from texar.torch.run.condition import Condition, Event, EventPoint
from texar.torch.run.executor_utils import Instance, OptionalDict, OptionalList
from texar.torch.run.metric import Metric

__all__ = [
    "make_deterministic",
    "Executor",
]

ActionFn = Callable[['Executor'], None]


def make_deterministic(seed: int = 19260817,
                       cudnn_deterministic: bool = False):
    r"""Make experiment deterministic by using specific random seeds across
    all frameworks and (optionally) use deterministic algorithms.

    Args:
        seed (int): The random seed to set.
        cudnn_deterministic (bool): If `True`, set CuDNN to use
            deterministic algorithms. Setting this to `True` can negatively
            impact performance, and might not be necessary for most cases.
            Defaults to `False`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Executor:
    r""":class:`Executor` is a substitute for the general training/evaluation
    loop. It is designed with the following goals in mind:

    1. **Minimize the amount of boilerplate code** that is essentially the same
       across all experiments.
    2. Provide **best practices** and hide hideous details from the user.
    3. Guarantee **reproducability** (runs with same config & seed always
       produces the same result) and **portability** (same code runs whether
       using GPU or not).
    4. Meanwhile, allowing **flexible configurations** and support
       user-overriden behaviors.

    Example:
        Here is a realistic training loop example using :class:`Executor`,
        showcasing the features built in. :class:`Executor` takes care of common
        training procedures including logging, checkpoint management, evaluation
        metrics, validation, and patience-based early-stopping.

        .. code-block:: python

            from texar.torch.run import *

            executor = Executor(
                model=model,
                train_data=datasets["train"],
                valid_data=datasets["dev"],
                test_data=datasets["test"],
                checkpoint_dir=args.save_dir,
                save_every=cond.validation(better=True),
                train_metrics=("loss", metric.RunningAverage(args.display_steps)),
                optimizer={"type": torch.optim.Adam},
                grad_clip=args.grad_clip,
                log_every=cond.iteration(args.display_steps),
                validate_every=cond.epoch(1),
                valid_metrics=[
                    metric.PearsonR(pred_name="preds"),
                    ("loss", metric.Average())],
                plateau_condition=[
                    cond.consecutive(cond.validation(better=False), 2)],
                action_on_plateau=[
                    action.early_stop(patience=2),
                    action.reset_params(),
                    action.scale_lr(0.8)],
                stop_training_on=cond.iteration(args.max_train_steps),
                test_mode='eval',
            )
            executor.train()

    .. |Metric| replace:: :class:`~texar.torch.run.metric.Metric`
    .. |Condition| replace:: :class:`~texar.torch.run.condition.Condition`
    .. |Optimizer| replace::
        :torch_docs:`torch.optim.Optimizer <optim.html#torch.optim.Optimizer>`
    .. |LRScheduler| replace::
        :torch_docs:`LRScheduler <optim.html#how-to-adjust-learning-rate>`
    .. |Action| replace:: :class:`~texar.torch.run.action.Action`
    .. |Event| replace:: :class:`~texar.torch.run.condition.Event`

    **Concepts**

    To make full use of :class:`Executor`, you'll need to understand the
    following concepts:

    - **Event:** Events are a set of pre-defined time spans within the training
      and evaluation loop. Common events include a training iteration
      (``Event.Iteration``), an epoch (``Event.Epoch``), or an entire validation
      (``Event.Validation``). Please refer to the enum class |Event| for the
      full list of events. The beginning and end of events are called **event
      points**.
    - **Condition:** |Condition|\ s are checks performed at the beginning or end of
      events. If a check passes, we say that the condition "triggers". To give
      a few examples:

      - :class:`cond.iteration(num_iters) <texar.torch.run.condition.iteration>`
        is checked only at the end of training iterations, and the check passes
        when the number of iterations passed equals :attr:`num_iters`. Thus, the
        condition triggers at the end of every :attr:`num_iters` iterations.
      - :class:`cond.validation(better=True)
        <texar.torch.run.condition.validation>` is checked only at the end of
        validations, and the check passes if the current validation result is
        better than the previous best. Thus, the condition triggers whenever
        a finished validation yields an improved result.
      - :class:`cond.time <texar.torch.run.condition.time>` is checked at the
        end of iterations, and at the beginning and end of training, validation,
        and testing. It checks whether the elapsed time has reached the
        specified duration, and triggers when it does.

      Custom conditions must subclass the |Condition| class.
    - **Action:** Actions are special callback functions that are called
      either at the beginning or end of an event (actions registered by
      :meth:`on_event`), or when conditions trigger (actions registered by
      :meth:`on`, and by specifying :attr:`save_every` and similar arguments).
      This is similar to hook functions that exists in common frameworks.
      Actions take a single argument -- the :class:`Executor` instance itself,
      and can perform any operations within.

      Custom actions can be simple functions, or subclass the |Action| class.
    - **Metric:** Metrics are used to evaluate the output of models. They are
      categorized into two classes:
      :class:`~texar.torch.run.metric.SimpleMetric` and
      :class:`~texar.torch.run.metric.StreamingMetric`. The only difference is
      that streaming metrics support incremental computation of metric values,
      so they can be used to aggregate results over the training set, or provide
      intermediate results on-the-fly.

      Custom metrics must subclass one of the above classes.

    **Customization**

    You can easily extend the :class:`Executor` class by subclassing it and
    overriding methods. Methods of interest include:

    - :meth:`_train_step`: Perform a single step of training (i.e. process a
      single batch, call :meth:`backward`, and potentially call optimizer
      updates). Takes the data batch as argument.
    - :meth:`_validate_step`: Perform a single step of validation. Takes the
      data batch as argument.
    - :meth:`_test_step`: Perform a single step of testing. Takes the data batch
      as argument.
    - :meth:`_train_loop`: Runs the entire training loop. Takes the data
      iterator as argument.
    - :meth:`_train_loop`: Runs the entire validation loop. Takes the data
      iterator as argument.
    - :meth:`_test_loop`: Runs the entire testing loop. Takes the data iterator
      as argument.

    You can also define custom events by writing a new enum class and modifying
    the :attr:`_EVENT_TYPES` attribute. Event points can be signaled by calling
    :meth:`_fire_event`. For example:

    .. code-block:: python

        class GANEvent(Enum):
            DiscriminatorUpdate = auto()
            GeneratorUpdate = auto()

        class GANExecutor(Executor):
            _EVENT_TYPES = (Event, GANEvent)

            def __init__(self, *args, optimizer_g, optimizer_d, **kwargs):
                kwargs["optimizer"] = {
                    "g": optimizer_g,
                    "d": optimizer_d,
                }
                super.__init__(*args, **kwargs)

            def _train_step(self, batch):
                self._fire_event(GANEvent.GeneratorUpdate, False)
                z = torch.randn(len(batch), args.latent_dim)
                fake_image = self.model.generator(z)
                logits = self.model.discriminator(fake_image)
                g_loss = F.binary_cross_entropy(logits, torch.ones(len(batch))
                g_loss.backward()
                self.optimizer["g"].step()
                self.optimizer["g"].zero_grad()
                self._fire_event(GANEvent.GeneratorUpdate, True)

                self._fire_event(GANEvent.DiscriminatorUpdate, False)
                real_logits = self.model.discriminator(batch.image)
                fake_logits = self.model.discriminator(fake_image.detach())
                real_loss = F.binary_cross_entropy(real_logits, torch.ones(len(batch)))
                fake_loss = F.binary_cross_entropy(fake_logits, torch.zeros(len(batch)))
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer["d"].step()
                self.optimizer["d"].zero_grad()
                self._fire_event(GANEvent.DiscriminatorUpdate, True)

                return {"g_loss": g_loss, "d_loss": d_loss}

    **Arguments**

    The constructor of :class:`Executor` takes many arguments, almost all of
    which are keyword-only and optional. Some arguments can take values of
    multiple types, either a single instance of a specific type, or a list or
    dictionary of values of that type.

    Arguments grouped by functions:

    - :ref:`General arguments <executor-general-args>`
    - :ref:`Arguments for checkpoint management <executor-checkpoint-args>`
    - :ref:`Arguments for training <executor-train-args>`
    - :ref:`Arguments for validation <executor-valid-args>`
    - :ref:`Arguments for testing <executor-test-args>`
    - :ref:`Arguments for logging <executor-log-args>`

    .. _executor-general-args:

    **General arguments:**

    `model`: :torch_nn:`Module`
        The model to train or evaluate. The model must be a subclass of
        :torch_nn:`Module`, with its :meth:`forward` method taking a single
        argument ``batch`` and returning a dictionary of :tensor:`Tensor`\ s:

        - The ``batch`` argument is of type :class:`~texar.torch.data.Batch`,
          which is the batch object produced by your provided dataset.
        - The returned dictionary must contain an entry named ``"loss"``, which
          will be used as the loss to backpropagate during training. You can
          also include other values and use them in metrics.

        If the model performs different routines during training and evaluation
        (for instance, a sequence-to-sequence model may train using
        teacher-forcing but evaluate using beam search-based inference), you can
        define another method :meth:`predict` following the same signature as
        :meth:`forward`. To use :meth:`predict` instead of :meth:`forward` in
        validation or testing, set :attr:`validate_mode` or :attr:`test_mode` to
        ``"predict"`` instead of ``"eval"``.

        If the model you use does not follow this convention, you will need to
        wrap the model in a new class. The following example demonstrates how to
        wrap :class:`~texar.torch.modules.XLNetRegressor`:

        .. code-block:: python

            class RegressorWrapper(tx.modules.XLNetRegressor):
                def forward(self, batch):
                    preds = super().forward(token_ids=batch.input_ids,
                                            segment_ids=batch.segment_ids,
                                            input_mask=batch.input_mask)
                    loss = (preds - batch.label_ids) ** 2
                    loss = loss.sum() / len(batch)
                    return {"loss": loss, "preds": preds}

    `train_data`: :class:`~texar.torch.data.DataBase`
        The dataset used during training. Must be specified for training.

    `valid_data`: :class:`~texar.torch.data.DataBase`
        The dataset used during validation. If not specified, you cannot perform
        validation during training (e.g., by setting :attr:`validate_every`).

    `test_data`: :class:`~texar.torch.data.DataBase`, or a list or dictionary
        The dataset(s) used during testing. If not specified, you cannot perform
        testing during training (e.g., by setting :attr:`test_every`).

    `batching_strategy`: :class:`~texar.torch.data.BatchingStrategy`
        The batching strategy to use for batching. This will be passed as the
        :attr:`batching_strategy` argument for
        :class:`~texar.torch.data.DataIterator` during training, and evaluation
        if the corresponding mode is set to ``"eval"``.

    `validate_mode`: str
        The evaluation mode for validation. Available choices are ``"eval"``
        and ``"predict"``. Defaults to ``"eval"``. When mode is set to
        ``"eval"``, :meth:`forward` method of the model will be called; when
        set to ``"predict"``, :meth:`predict` method of the model will be
        called.

    `test_mode`: str
        The evaluation mode for testing. Available choices are ``"eval"``
        and ``"predict"``. Defaults to ``"predict"``.

    `device`: ``torch.device``
        The device on which the model and data should be placed. Defaults to
        `None`, in which case GPUs will be used if available.

    .. _executor-checkpoint-args:

    **Arguments for checkpoint management:**

    `checkpoint_dir`: str
        Path to the directory for storing checkpoints. If not specified, you
        cannot save/load the model during training (e.g., by setting
        :attr:`save_every` or using
        :class:`~texar.torch.run.action.reset_params`).

    `max_to_keep`: int
        The maximum number of checkpoints to keep in the checkpoint directory.
        When the number of checkpoints exceed this limit, the oldest one will be
        removed. If `None`, no such limit is imposed. Defaults to `None`.

        .. note::
            Be careful when saving periodic snapshots along with the best
            performing checkpoint. Periodic snapshots might overwrite best
            models if checkpoint limit is exceeded.

            A better workaround is to only save periodic snapshots with the
            built-in mechanism, and register a custom action for saving a single
            best performing checkpoint:

            .. code-block:: python

                # Don't
                executor = Executor(
                    save_every=[cond.epoch(1), cond.validation(better=True)],
                    max_to_keep=3, ...)

                # Do
                executor = Executor(
                    save_every=cond.epoch(1),
                    max_to_keep=3, ...)

                @executor.on(cond.validation(better=True))
                def save_best_model(executor):
                    executor.save(path=another_directory, max_to_keep=1)

    `save_every`: |Condition|, or a list
        Conditions that, when triggered, saves the model.

        In the following example, the model will be saved every 1000 iterations,
        or whenever validation results improve.

        .. code-block:: python

            save_every=[cond.validation(better=True), cond.iteration(1000)]

    `save_training_state`: bool
        If `False`, only save model parameters in checkpoint. If `True`, also
        save optimizer and scheduler states, along with random number generator
        states from Python, NumPy, and PyTorch. Defaults to `True`.

    .. _executor-train-args:

    **Arguments for training:**

    .. _train-metrics:

    `train_metrics`: |Metric|, or a list or dictionary
        The metrics computed over the training set. In case of multiple metrics,
        two sets of metric values will be compared in the provided order.

        For example, if two metrics `f1` (:class:`~texar.torch.run.metric.F1`)
        and `loss` (:class:`~texar.torch.run.metric.Average`) are defined (in
        this order), when comparing two sets of values, the one with a higher
        `f1` is considered better. If the two sets have the same `f1` value, the
        one with a lower `loss` is considered better.

        Acceptable values include:

        - A single |Metric|, or a list of |Metric|\ s. These metrics will be
          automatically named when they're logged.
        - A tuple of (``str``, |Metric|), or a list of this. These metrics are
          explicitly named according to the provided strings. Note that names
          must be unique.
        - An :class:`~collections.OrderedDict` mapping names to metrics. Note
          that a plain (unordered) dictionary is not accepted.

        .. note::
            Metrics that are logged will be evaluated once every time logging
            is performed. For efficiency considerations, such metrics should be
            :class:`~texar.torch.run.metric.StreamingMetric`\ s. Please take
            extra care when implementing your own metrics.

    `optimizer`: |Optimizer|, or a dictionary of hyperparameters
        The optimizer used during training. This can be a |Optimizer| instance,
        or a dictionary of hyperparameters that will be passed into
        :meth:`texar.torch.utils.get_instance`. Must be specified for training.

    `lr_scheduler`: |LRScheduler|, or a dictionary of hyperparameters, optional
        The learning rate scheduler used during training. This can be an
        |LRScheduler| instance, or a dictionary of hyperparameters that will be
        passed into :meth:`texar.torch.utils.get_instance`.

    `stop_training_on`: |Condition|, or a list
        Conditions that, when triggered, will stop training.

        In the following example, training will be terminated after 5 epochs or
        20000 iterations, whichever comes first:

        .. code-block:: python

            stop_training_on=[cond.epoch(5), cond.iteration(20000)]

    `num_iters_per_update`: int
        Number of iterations to run before performing a parameter update. When
        this value is greater than 1, the loss is scaled by its reciprocal.
        Defaults to 1, in which case the parameters are updated after each
        ``.backward()`` call.

        This can be used to accumulate gradients across multiple batches, in
        order to simulate the effect of using a large batch size on a machine
        with limited memory.

    `grad_clip`: float, optional
        Maximum norm of the gradients. Please refer to
        :torch_docs:`nn.utils.clip_grad_norm_ <nn.html#torch.nn.utils.clip_grad_norm_>`
        for details. Defaults to `None`, i.e. no clipping.

    .. _executor-valid-args:

    **Arguments for validation:**

    `valid_metrics`: |Metric|, or a list or dictionary
        The metrics computed over the validation set. Please see
        :ref:`train_metrics <train-metrics>` for details.

    `validate_every`: |Condition|, or a list
        Conditions that, when triggered, performs validation.

        In the following example, the model will be validated once per epoch.

        .. code-block:: python

            validate_every=cond.epoch(1)

    `plateau_condition`: |Condition|, or a list
        Conditions that, when triggered, indicates that training has reached
        a plateau, i.e., the model has stopped improving.

        In the following example, we consider that training has reached a
        plateau if validation metrics have not improved for 3 consecutive
        validations.

        .. code-block:: python

            plateau_condition=cond.consecutive(cond.validation(better=False))

    `action_on_plateau`: |Action|, or a list
        Actions that will be called when training has reached a plateau.

        In the following example, we perform patience-based early-stopping when
        reaching plateaus. A patience of 2 means training will be terminated
        after plateau is reached twice. We also scale the learning rate by 0.8,
        and reset the model & optimizer parameters to the previous best
        checkpoint.

        .. code-block:: python

            action_on_plateau=[
                action.reset_params(), action.scale_lr(0.8),
                action.early_stop(patience=2)]

    .. _executor-test-args:

    **Arguments for testing:**

    `test_metrics`: |Metric|, or a list or dictionary
        The metrics computed over the test set. Please see
        :ref:`train_metrics <train-metrics>` for details. :meth:`test` can only
        be called if :attr:`test_metrics` is not `None`.

        .. note::
             :attr:`valid_metrics` will be automatically shared with
             :attr:`test_metrics` if:

             1. :attr:`test_metrics` is `None`;
             2. :attr:`validate_mode` is the same as :attr:`test_mode`.

    `test_every`: |Condition|, or a list
        Conditions that, when triggered, performs testing.

        In the following example, the model will be tested whenever validation
        results improve.

        .. code-block:: python

            test_every=cond.validation(better=True)

    .. _executor-log-args:

    **Arguments for logging:**

    `log_every`: |Condition|, or a list
        Conditions that, when triggered, performs logging.

        In the following example, a log will be printed every 100 iterations,
        and after every epoch.

        .. code-block:: python

            log_every=[cond.iteration(100), cond.epoch()]

    `log_destination`: ``str``, IO object, or a list
        Logging destinations. Acceptable values include:

        - A string, denoting the path to a log file. The file will be opened in
          "append" (``"a"``) mode.
        - An IO object. This can be an opened file, ``sys.stdout``, or any other
          file-like object.
        - A list, with each element being one of the above.

        When writing to a file, special syntax for terminals (e.g. color codes)
        are emitted. Also, live progress is not written to files.

        By default, the log is only written to ``sys.stdout``.

    .. _log-format:

    `log_format`: ``str``
        The format string for logs during training.

        The format string follows the syntax of Python format strings. The
        status variables you can reference include:

        - ``epoch`` (int): The current epoch.
        - ``iteration`` (int): The current iteration.
        - ``progress`` (float): The epoch progress represented in percentage,
          i.e. a floating-point number between 0 and 100. It should be noted
          that progress may not be accurate, and may not be available if the
          data is loaded lazily.
        - ``speed`` (float): Average number of data examples processed per
          second. It should be noted that speed may not be accurate.
        - ``time``: The current date and time. Time format can be set using the
          "format spec" syntax of Python format strings, i.e.:
          ``{time:%H:%M:%S}`` prints time in the format ``08:26:03``. If time
          format is not specified, it is equivalent to
          ``{time:%Y-%m-%d %H:%M:%S}``, which corresponds to the format
          ``2018-07-06 08:26:03``. For more information on time formatting,
          please refer to documentation for Python built-in function
          :meth:`date.strftime`.
        - ``metric``: An aggregated representation of all metrics, in the
          format of ``<name1>: <value1>, <name2>: <value2>, ...``. The format
          spec for ``metric`` will be applied to all metrics whose value
          supports such format spec.
          For more fine grained control over metric formatting, use the
          following methods.
        - ``metric.<name>``: Value of the metric under the specified name
          ``<name>``.
        - ``<name>``: Value of the metric under the specified name ``<name>``.

          .. note::
              The metric can only be looked-up if its name does not coincide
              with built-in status variables. For instance, a metric named
              "`loss`" can be looked-up by ``loss`` and ``metric.loss``, but a
              metric named "`time`" can only be looked-up by ``metric.time``.

        The default format string is:

        .. code-block::

            {time} : Epoch {epoch} @ {iteration}it ({progress}%, {speed}), {{{metric:.3f}}}

        which produces logs similar to:

        .. code-block::

            2019-08-05 11:46:26 : Epoch 1 @ 800it (20.3%, 16.14ex/s), {loss: 0.358, Accuracy: 0.567}

    `eval_log_format`: str
        The format string for logs during evaluation (i.e. validation and
        testing). Please refer to :ref:`log_format <log-format>` for details of
        the format string.

        The default format string is:

        .. code-block::

            {time} : Epoch {epoch}, {split} result = {{{metric:.3f}}}

        which produces logs similar to:

        .. code-block::

            2019-08-05 11:36:53 : Epoch 6, valid result = {PearsonR: 0.918, loss: 0.363}

    `print_configs_to_log`: bool
        If `True`, model architecture and :class:`Executor` configurations will
        logged in a readable format. Defaults to `True`.

    `show_live_progress`: bool
        If `True`, live progress will be shown. The specified format string will
        be shown similar to a sticky progress bar at the bottom of the terminal
        window, and updated after each iteration.

        Note that live progress is only shown on terminals. It will not be
        printed to log files.

        .. warning::
            This may incur extra overhead because an update requires
            re-evaluating metrics. Make sure that all metrics logged are
            :class:`~texar.torch.run.metric.StreamingMetric`\ s. You can also
            explicitly log only streaming metrics, disable live progress for
            evaluation, or use a separate template???
    """
    _EVENT_TYPES = (Event,)
    _CHECKPOINT_METAINFO_FILE = "checkpoint.meta-info"
    _CHECKPOINT_EXTENSION = ".pt"
    _defaults: Dict[str, Any] = {
        "log_format": "{time} : Epoch {epoch} @ {iteration}it "
                      "({progress}%, {speed}), {{{metric:.3f}}}",
        "eval_log_format": "{time} : Epoch {epoch}, "
                           "{split} result = {{{metric:.3f}}}",
        "log_destination": [sys.stdout],
    }

    # TODO: Add loss as a special default metric? Otherwise would have to use
    #   RunningAverage with display steps

    # TODO: Prevent the same action triggering twice without any other steps.

    _hooks: Dict[EventPoint, Dict[Optional[Condition], List[ActionFn]]]

    status: utils.TrainingStatus

    def __init__(self, model: nn.Module,
                 *,
                 train_data: Optional[DataBase] = None,
                 valid_data: Optional[DataBase] = None,
                 test_data: OptionalDict[DataBase] = None,
                 batching_strategy: Optional[BatchingStrategy] = None,
                 device: Optional[torch.device] = None,
                 # Checkpoint
                 checkpoint_dir: Optional[str] = None,
                 max_to_keep: Optional[int] = None,
                 save_every: OptionalList[Condition] = None,
                 save_training_state: bool = True,
                 # Training
                 train_metrics: OptionalDict[Metric] = None,
                 optimizer: Optional[Instance[Optimizer]] = None,
                 lr_scheduler: Optional[Instance[LRScheduler]] = None,
                 stop_training_on: OptionalList[Metric] = None,
                 num_iters_per_update: int = 1,
                 grad_clip: Optional[float] = None,
                 # Validation
                 valid_metrics: OptionalDict[Metric] = None,
                 validate_every: OptionalList[Condition] = None,
                 plateau_condition: OptionalList[Condition] = None,
                 action_on_plateau: OptionalList[ActionFn] = None,
                 validate_mode: str = 'eval',
                 # Testing
                 test_metrics: OptionalDict[Metric] = None,
                 test_every: OptionalList[Condition] = None,
                 test_mode: str = 'predict',
                 # Logging
                 log_every: OptionalList[Condition] = None,
                 log_format: Optional[str] = None,
                 log_destination: OptionalList[Union[str, IO[str]]] = None,
                 print_configs_to_log: bool = True,
                 eval_log_format: Optional[str] = None,
                 show_live_progress: bool = False,
                 # Tensorboard
                 # pylint: disable=unused-argument
                 tensorboard_log_dir: Optional[str] = None,
                 write_summary_every: OptionalList[Condition] = None
                 # pylint: enable=unused-argument
                 ):

        # TODO: Add support for Tensorboard.

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = utils.to_dict(test_data, default_name="test")
        self.batching_strategy = batching_strategy

        # Device placement
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(torch.cuda.current_device())
            else:
                device = torch.device('cpu')
        self.device = device
        self.model.to(device)

        # Logging
        self._log_conditions = utils.to_list(log_every)
        self._print_configs_to_log = print_configs_to_log
        self._show_progress_bar = show_live_progress

        self.log_format: str = log_format or self._defaults["log_format"]
        self.eval_log_format: str = (
                eval_log_format or self._defaults["eval_log_format"])

        # Checkpoint management
        self.checkpoint_dir = (Path(checkpoint_dir)
                               if checkpoint_dir is not None else None)
        self.max_to_keep = max_to_keep
        self._save_conditions = utils.to_list(save_every)
        self._save_training_state = save_training_state

        self._directory_exists = False
        if self.checkpoint_dir is not None:
            if not self.checkpoint_dir.exists():
                self.checkpoint_dir.mkdir(parents=True)
            else:
                self._directory_exists = True

        # TODO: Close files somewhere? Maybe `atexit.register`?
        # Create logging files after checkpoint directory is created.
        self.log_destination: List[IO[str]] = [
            # We append to the logs to prevent accidentally overwriting previous
            # logs. To force overwrite, pass IO objects instead of file paths.
            open(dest, "a") if isinstance(dest, (str, Path)) else dest  # type: ignore
            for dest in utils.to_list(  # type: ignore
                log_destination or self._defaults["log_destination"])]

        # Training loop
        self.train_metrics = utils.to_metric_dict(train_metrics)
        self.optimizer = utils.to_instance(
            Optimizer, optimizer, ["torch.optim", "texar.core"],
            extra_kwargs={"params": self.model.parameters()})
        self.lr_scheduler = utils.to_instance(
            LRScheduler, lr_scheduler,
            ["torch.optim.lr_scheduler", "texar.core"],
            extra_kwargs={"optimizer": self.optimizer})
        self._stop_training_conditions = utils.to_list(stop_training_on)
        self.num_iters_per_update = num_iters_per_update
        self.grad_clip = grad_clip
        self._should_terminate = False

        # Validation
        self.valid_metrics = utils.to_metric_dict(valid_metrics)
        self._valid_conditions = utils.to_list(validate_every)
        self.validate_mode = validate_mode
        self._plateau_conditions = utils.to_list(plateau_condition)
        self._actions_on_plateau = utils.to_list(action_on_plateau)

        # Testing
        if (test_metrics is None and valid_metrics is not None and
                validate_mode == test_mode):
            self.test_metrics = copy.deepcopy(self.valid_metrics)
        else:
            self.test_metrics = utils.to_metric_dict(test_metrics)
        self._test_conditions = utils.to_list(test_every)
        self.test_mode = test_mode

        # Initialize hooks.
        self._hooks = {
            (event, point): defaultdict(list)
            for event in Event for point in (False, True)}

        self.status = {
            "epoch": 0,
            "iteration": 0,
            "split": "train",
            "metric": self.train_metrics,
            "eval_metric": self.valid_metrics,
        }

        # TODO: Move trackers to another process, as in tqdm? Refer to `tqdm.TMonitor`
        self._train_tracker = utils.ProgressTracker()
        self._valid_tracker = utils.ProgressTracker()
        self._test_tracker = utils.ProgressTracker()

        # Register events & actions.
        train_logging_hook = self._create_logging_hook(
            self.log_format, self.train_metrics, self._train_tracker)
        for cond in self._log_conditions:
            self._register_action(cond, train_logging_hook)
        self._register_hook((Event.Validation, True), self._create_logging_hook(
            self.eval_log_format, self.valid_metrics, self._valid_tracker))
        self._register_hook((Event.Testing, True), self._create_logging_hook(
            self.eval_log_format, self.test_metrics, self._test_tracker))

        for cond in self._valid_conditions:
            self._register_action(cond, self._validate.__func__)  # type: ignore
        for cond in self._test_conditions:
            self._register_action(cond, self.test.__func__)  # type: ignore
        for cond in self._save_conditions:
            self._register_action(cond, self.save.__func__)  # type: ignore

        for cond in self._plateau_conditions:
            for action in self._actions_on_plateau:
                self._register_action(cond, action)

        for cond in self._stop_training_conditions:
            self._register_action(cond, self.terminate.__func__)  # type: ignore

    def write_log(self, log_str: str, *, mode: str = "info",
                  skip_tty: bool = False):
        if mode == "log":
            pass
        elif mode == "info":
            time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            pass
            log_str = utils.color(f"INFO {time_str} : ", "green") + log_str
        elif mode == "warning":
            time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            log_str = f"WARNING {time_str} : {log_str}"
            log_str = utils.color(log_str, "red")
        else:
            raise ValueError(f"Invalid logging mode {mode}")
        plain_str = re.sub(r"\033\[\d{1,2}[Km]", "", log_str)
        for dest in self.log_destination:
            if dest.isatty():
                if skip_tty:
                    continue
                dest.write(log_str)
            else:
                # Erase color codes if the destination is not a terminal.
                dest.write(plain_str)
            dest.write("\n")
            dest.flush()

    # pylint: disable=unused-argument,no-self-use,function-redefined

    @overload
    def on(self, cond: Condition) -> Callable[[ActionFn], ActionFn]:
        ...

    @overload
    def on(self, cond: Condition, func: ActionFn) -> 'Executor':
        ...

    def on(self, cond: Condition, func=None):
        r"""Register a function as an action triggered on a condition. For
        example:

        .. code-block:: python

            executor = Executor(...)

            @executor.on(cond.iteration(10, mode="valid"))
            def log_during_validation(executor):
                logging.info("Validation iter %d", executor.status.iteration)

        The action function takes exactly one argument: the executor instance
        itself.

        Args:
            cond: The condition that will call the function when triggered. Must
                be of type :class:`Condition`.
            func (optional): The function to register. If not `None`, the
                function will be registered; if `None`, a decorator will be
                returned.

        :returns:
            - If :attr:`func` is `None`, this method will return a decorator to
              wrap around the function to register as hook.
            - If :attr:`func` is not `None`, this method will return the
              :class:`Executor` itself, allowing chained calls.
        """
        if not isinstance(cond, Condition):
            raise ValueError(f"Invalid condition {cond}")

        if func is not None:
            self._register_action(cond, func)
            return self

        def wrapper(f):
            self._register_action(cond, f)
            return f

        return wrapper

    @overload
    def on_event(self, event: Event, point: str = 'end') \
            -> Callable[[ActionFn], ActionFn]:
        ...

    @overload
    def on_event(self, event: Event, point: str, func: ActionFn) -> 'Executor':
        ...

    def on_event(self, event, point='end', func=None):
        r"""Register a function as an action triggered on an event point. For
        example:

        .. code-block:: python

            executor = Executor(...)

            @executor.on_event(Event.Epoch, 'end')
            def log_at_end_of_epoch(executor):
                logging.info("Epoch %d done", executor.status.epoch)

        The action function takes exactly one argument: the executor instance
        itself.

        Args:
            event: The event to hook on. Must be an enum value from
                :class:`~Event`.
            point (str): The point of event to hook on. Supported values are
                ``"begin"`` and ``"end"``. Defaults to ``"end"``.
            func (optional): The function to register. If not `None`, the
                function will be registered; if `None`, a decorator will be
                returned.

        :returns:
            - If :attr:`func` is `None`, this method will return a decorator to
              wrap around the function to register as hook.
            - If :attr:`func` is not `None`, this method will return the
              :class:`Executor` itself, allowing chained calls.
        """
        if (not isinstance(event, self._EVENT_TYPES) or
                point not in ('begin', 'end')):
            raise ValueError(f"Invalid event point ({event}, {point})")
        end = (point == 'end')

        if func is not None:
            self._register_hook((event, end), func)
            return self

        def wrapper(f):
            self._register_hook((event, end), f)
            return f

        return wrapper

    # pylint: enable=unused-argument,no-self-use,function-redefined

    def save(self):
        # TODO: support saving to custom directory.

        # Load the checkpoint meta-info file.
        meta_path = self.checkpoint_dir / self._CHECKPOINT_METAINFO_FILE
        if meta_path.exists():
            try:
                with meta_path.open("rb") as f:
                    meta_dict = pickle.load(f)
            except (EOFError, IOError):
                meta_dict = {}
        else:
            meta_path.touch()
            meta_dict = {}

        # Remove earliest checkpoints if exceeds `max_to_keep`.
        if self.max_to_keep is not None:
            while len(meta_dict) >= self.max_to_keep:
                checkpoint_name = min(
                    meta_dict, key=lambda name: meta_dict[name]["timestamp"])
                (self.checkpoint_dir / checkpoint_name).unlink()
                del meta_dict[checkpoint_name]
                self.write_log(f"Previous checkpoint {checkpoint_name} removed "
                               f"due to `max_to_keep`(={self.max_to_keep}) "
                               f"limit", mode="info")

        timestamp = time.time()
        checkpoint_name = str(timestamp) + self._CHECKPOINT_EXTENSION
        path = self.checkpoint_dir / checkpoint_name
        if path.exists():
            idx = 0
            while True:
                checkpoint_name = (
                        str(timestamp) + f".{idx}" + self._CHECKPOINT_EXTENSION)
                path = self.checkpoint_dir / checkpoint_name
                if not path.exists():
                    break
        if self._save_training_state and self.optimizer is not None:
            train_state = utils.SavedTrainingState(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=(self.lr_scheduler.state_dict()
                           if self.lr_scheduler is not None else None),
                system_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                torch_rng=torch.random.get_rng_state(),
            )
            torch.save(train_state, str(path))
        else:
            torch.save(self.model.state_dict(), str(path))
        meta_dict[checkpoint_name] = {
            "status": self.status,
            "timestamp": timestamp,
        }
        with meta_path.open("wb") as f:
            pickle.dump(meta_dict, f)

        self.write_log(f"Current checkpoint saved to {path}", mode="info")

    def load(self, path: Optional[str] = None,
             load_training_state: bool = True,
             allow_failure: bool = False) -> Optional[Path]:
        r"""Load a previous model checkpoint from file.

        Args:
            path (str, optional): Path to a specific checkpoint or a checkpoint
                directory. If a directory is specified, the most recent
                checkpoint in the directory is loaded. If `None`,
                :attr:`checkpoint_dir` in the constructor arguments will be
                used. Defaults to `None`.
            load_training_state (bool): If `True`, will load entire training
                state from checkpoint (if the checkpoint contains training
                state). Otherwise, just load model weights. Defaults to `True`.
            allow_failure (bool): If `True`, no exceptions will be raised if
                no checkpoints were found. Defaults to `False`. Note that
                exceptions are still raised if the provided :attr:`path` does
                not exist, or the selected checkpoint is corrupted.

        Returns:
            Path of the loaded checkpoint, or `None` if load failed.
        """
        if path is not None:
            ckpt_path = Path(path)
        elif self.checkpoint_dir is not None:
            ckpt_path = self.checkpoint_dir
        else:
            raise ValueError(
                "`path` must be specified when `checkpoint_dir` is `None`")
        if ckpt_path.is_dir():
            try:
                meta_path = ckpt_path / self._CHECKPOINT_METAINFO_FILE
                if meta_path.exists():
                    with meta_path.open("rb") as f:
                        meta_dict = pickle.load(f)
                    best_metric, best_m_time, best_ckpt_name = max(
                        (utils.MetricList(info["status"]["eval_metric"]),
                         info["timestamp"], name)
                        for name, info in meta_dict.items())
                    status = meta_dict[best_ckpt_name]["status"]
                    ckpt_path = self.checkpoint_dir / best_ckpt_name
                    metric_vals = [(name, best_metric.values[name])
                                   for name in best_metric.metrics]
                    metric_str = ", ".join(
                        f"{name}: " + (
                            f"{value:.3f}" if isinstance(value, float)
                            else f"{value}")
                        for name, value in metric_vals)
                    time_str = datetime.fromtimestamp(best_m_time).strftime(
                        "%Y-%m-%d %H:%M:%S")
                    load_info_str = (f"saved at: {time_str}, "
                                     f"meta-info: epoch={status['epoch']}, "
                                     f"iteration={status['iteration']}, "
                                     f"metrics={{{metric_str}}}")
                else:
                    m_time, ckpt_path = max(
                        (name.stat().st_mtime, name)
                        for name in ckpt_path.iterdir()
                        if name.suffix == self._CHECKPOINT_EXTENSION)
                    time_str = datetime.fromtimestamp(m_time).strftime(
                        "%Y-%m-%d %H:%M:%S")
                    load_info_str = f"saved at: {time_str}"
                    self.write_log(
                        "Checkpoint meta-info not found. Will load the most "
                        "recent checkpoint in directory", mode="warning")
            except (EOFError, IOError, ValueError):
                # EOFError, IOError: pickle.load
                # ValueError: max() arg is an empty sequence
                if allow_failure:
                    return None
                raise
        else:
            load_info_str = ""

        checkpoint = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(checkpoint, utils.SavedTrainingState):
            self.model.load_state_dict(checkpoint.model)
            if load_training_state:
                # TODO: Also somehow save/load data iterator state?
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint.optimizer)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint.scheduler)
                random.setstate(checkpoint.system_rng)
                np.random.set_state(checkpoint.numpy_rng)
                torch.random.set_rng_state(checkpoint.torch_rng.cpu())
        else:
            self.model.load_state_dict(checkpoint)

        if load_info_str != "":
            self.write_log(f"Checkpoint ({load_info_str}) "
                           f"loaded from {ckpt_path}.", mode="info")
        else:
            self.write_log(f"Checkpoint loaded from {ckpt_path}.", mode="info")
        return ckpt_path

    def terminate(self) -> None:
        r"""Terminate training. This method is intended to be called within
        actions, an example use case would be to implement a custom
        early-stopping mechanism.

        It is guaranteed that no other event points will be fired once
        :meth:`terminate` is called. However, conditions and actions under the
        same event point is still called.
        """
        self._should_terminate = True

    def train(self):
        r"""Start the training loop.
        """
        if self._directory_exists:
            self.write_log(
                f"Specified checkpoint directory '{self.checkpoint_dir}' "
                f"exists, previous checkpoints might be erased", mode="warning")

        if len(self._stop_training_conditions) == 0:
            self.write_log(
                "`stop_training_on` is not configured. Unless an event action "
                "calls `executor.terminate()`, training will run indefinitely.",
                mode='warning')

        # Initialize optimizer.
        if self.optimizer is None:
            raise ValueError("Optimizer is not specified")
        self.optimizer.zero_grad()

        # Initialize dataset.
        if self.train_data is None:
            raise ValueError("No training dataset is specified")
        self.train_data.to(self.device)
        iterator = DataIterator(self.train_data, self.batching_strategy)
        if len(self._valid_conditions) > 0:
            if self.valid_data is None:
                raise ValueError("Validation will be performed, but no "
                                 "validation dataset is specified.")
            self.valid_data.to(self.device)

        self.status["split"] = "train"

        self._fire_event(Event.Training, False)
        self.write_log("Training started", mode="info")

        if self._print_configs_to_log:
            # TODO: Also somehow gather training settings?
            self.write_log(f"Model architecture:\n{self.model!r}", mode="info")

        self.model.train()

        # Main training loop.
        try:
            self._train_loop(iterator)
        except utils.TerminateExecution:
            self.write_log("Training terminated", mode='info')
        self._fire_event(Event.Training, True)

    def test(self, dataset: OptionalDict[DataBase] = None):
        r"""Start the test loop.

        Args:
            dataset (optional): The dataset(s) to test on. Acceptable values
                include:

                - A single :attr:`~texar.torch.data.DataBase` instance.
                - A list of :attr:`~texar.torch.data.DataBase` instances.
                - A dictionary mapping names to
                  :attr:`~texar.torch.data.DataBase` instances.

                If `None`, :attr:`test_data` from the constructor arguments is
                used. Defaults to `None`.
        """
        if dataset is None and self.test_data is None:
            raise ValueError("No testing dataset is specified")
        if len(self.test_metrics) == 0:
            raise ValueError(
                "No testing metric is specified. Validation metrics are not "
                "used due to different modes for validation and test")
        datasets = (self.test_data if dataset is None
                    else utils.to_dict(dataset, default_name="test"))

        model_mode = self.model.training
        self.model.train(self.test_mode == "train")
        with torch.no_grad():
            for name, data in datasets.items():
                try:
                    self.status["split"] = name

                    # Initialize metrics.
                    for metric in self.test_metrics.values():
                        metric.reset()

                    self._fire_event(Event.Testing, False)
                    data.to(self.device)
                    if self.test_mode == "eval":
                        iterator = DataIterator(data, self.batching_strategy)
                    else:
                        iterator = DataIterator(data)
                    try:
                        data_size: Optional[int] = len(data)
                    except TypeError:
                        data_size = None
                    self._test_tracker.reset(data_size)

                    self._test_loop(iterator)
                except utils.TerminateExecution:
                    self.write_log(
                        "Testing terminated. This is likely unintended. Please "
                        "check your custom actions.", mode='warning')
                self._fire_event(Event.Testing, True)
        self.model.train(model_mode)

    def _create_logging_hook(self, format_str: str,
                             metrics: 'OrderedDict[str, Metric]',
                             tracker: utils.ProgressTracker) -> ActionFn:
        r"""Given a logging format string, create a function that takes the
        executor instance as argument and returns the logging string.

        Args:
            format_str (str): The logging format string.
            metrics: The metrics dictionary that will be used in the logging
                hook function.
            tracker: The progress tracker that will be used in the logging hook
                function.

        Returns:
            A hook function to print logs given the format string. Note that
            the hook function can accept additional arguments to pass to
            :meth:`executor.write_log`, allowing it to be used in combination
            with :meth:`functools.partial`.
        """
        format_var_regex = re.compile(r"{([a-zA-Z_0-9]+)(:([^{}]+))?}")
        format_vars: Dict[str, Optional[str]] = {
            match.group(1): match.group(3)
            for match in format_var_regex.finditer(format_str)
        }

        # Set default time format.
        if "time" in format_vars and format_vars["time"] is None:
            format_vars["time"] = "%Y-%m-%d %H:%M:%S"
            format_str = re.sub(r"{time(:([^{}]+))?}", "{time}", format_str)

        # Set metric print formats for aggregated format variable.
        if "metric" in format_vars and format_vars["metric"] is not None:
            # Check which metrics can be printed with specified format.
            fmt_metrics: Set[str] = set()
            metric_format = format_vars["metric"]
            for name, metric in metrics.items():
                try:
                    metric_format.format(metric.value())
                    fmt_metrics.add(name)
                except ValueError:
                    pass
            metric_format_parts = []
            for name in metrics:
                metric_format_parts.append(
                    f"{name}: {{{name}"
                    f"{':' + metric_format if name in fmt_metrics else ''}}}")
            metric_format_str = ', '.join(metric_format_parts)
            format_vars["metric"] = metric_format_str
            format_str = re.sub(r"{metric(:([^{}]+))?}", "{metric}", format_str)

        # Gather metrics represented as "name" or "metric.name".
        metrics_to_print: Set[str] = set()
        for name in format_vars:
            if name in self.status or name in ["time", "progress", "speed"]:
                # built-in name
                pass
            elif name in metrics:
                metrics_to_print.add(name)
            else:
                raise ValueError(
                    f"Invalid status variable name '{name}' in format string")

        format_str_wo_progress = re.sub(
            r"{progress(:([^{}]+))?}", "{progress}", format_str)

        def log_fn(executor: 'Executor', **log_kwargs):
            format_args = executor.status.copy()
            cur_format_str = format_str
            if "time" in format_vars:
                format_args["time"] = time.strftime(format_vars["time"])
            if "metric" in format_vars:
                metric_vals = {
                    name: metric.value() for name, metric in metrics.items()}
                metric_str = format_vars["metric"].format(**metric_vals)
                format_args["metric"] = metric_str
            if "speed" in format_vars:
                format_args["speed"] = tracker.speed()
            if "progress" in format_vars:
                progress = tracker.progress()
                if progress is not None:
                    if format_vars["progress"] is None:
                        progress = round(progress, 1)
                    format_args["progress"] = progress
                else:
                    cur_format_str = format_str_wo_progress
                    format_args["progress"] = "unknown"
            format_args.update({
                name: metrics[name].value() for name in metrics_to_print})
            log_str = cur_format_str.format(**format_args)

            self.write_log(log_str, mode="log", **log_kwargs)

        return log_fn

    def _register_action(self, cond: Condition, action: ActionFn):
        for event_point in cond.hooks:
            self._register_hook(event_point, action, cond)

    def _register_hook(self, event_point: EventPoint, action: ActionFn,
                       cond: Optional[Condition] = None):
        try:
            self._hooks[event_point][cond].append(action)
        except KeyError:
            raise ValueError(
                f"Specified hook point {event_point} is invalid") from None

    def _fire_event(self, event: Event, end: bool):
        r"""Signal the beginning or end of an event. Internally, this is where
        conditions are checked and actions are executed.

        Args:
            event: The |Event| to fire.
            end: If `True`, the fired event point is the end of :attr:`event`.
                If `False`, the fired event point is the beginning of
                :attr:`event`.

        :raises: If any triggered action calls :meth:`terminate`,
            :exc:`~texar.torch.run.executor_utils.TerminateException` is thrown
            after all conditions are checked and actions executed.
        """
        for cond, actions in self._hooks[(event, end)].items():
            # If condition is `None` (raw function hooks), action always
            # triggers.
            if cond is None or cond.hooks[(event, end)](self):
                for action in actions:
                    action(self)
        if self._should_terminate:
            self._should_terminate = False
            raise utils.TerminateExecution

    def _validate_step(self, batch: Batch):
        r"""Perform one step of validation, i.e., perform a forward pass (or
        decoding, depending on :attr:`validate_mode`) for a single batch.

        Args:
            batch: The batch to validate on.

        Returns:
            The dictionary containing values returned by the model. This is used
            to compute metrics.
        """
        if self.validate_mode == 'predict':
            return_dict = self.model.predict(batch)
        else:
            return_dict = self.model(batch)
        return return_dict

    def _test_step(self, batch: Batch):
        r"""Perform one step of testing, i.e., perform a forward pass (or
        decoding, depending on :attr:`test_mode`) for a single batch.

        Args:
            batch: The batch to test on.

        Returns:
            The dictionary containing values returned by the model. This is used
            to compute metrics.
        """
        if self.test_mode == 'predict':
            return_dict = self.model.predict(batch)
        else:
            return_dict = self.model(batch)
        return return_dict

    def _train_step(self, batch: Batch):
        r"""Perform one step of training, i.e., perform a forward and backward
        pass for a single batch. Parameter updates should also be performed when
        necessary.

        Args:
            batch: The batch to train on.

        Returns:
            The dictionary containing values returned by the model. This is used
            to compute metrics.
        """
        return_dict = self.model(batch)
        try:
            loss = return_dict['loss']
        except KeyError:
            raise ValueError("Return dictionary from model does not "
                             "contain 'loss' entry")
        loss /= self.num_iters_per_update
        loss.backward()
        if (self.num_iters_per_update == 1 or
                self.status["iteration"] % self.num_iters_per_update == 0):
            self._fire_event(Event.ParameterUpdate, False)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # type: ignore
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # type: ignore
            self.optimizer.zero_grad()  # type: ignore
            self._fire_event(Event.ParameterUpdate, True)
        return return_dict

    def _train_loop(self, iterator: DataIterator) -> None:
        r"""Run the entire training loop given the data iterator.

        Args:
            iterator: The iterator over the training data.
        """
        epoch = 0
        iteration = 0

        while True:
            try:
                data_size: Optional[int] = len(self.train_data)
            except TypeError:
                data_size = None
            self._train_tracker.reset(data_size)

            epoch += 1
            self.status["epoch"] = epoch
            # Initialize metrics.
            for metric in self.train_metrics.values():
                metric.reset()

            self._fire_event(Event.Epoch, False)

            for batch in iterator:
                self._fire_event(Event.Iteration, False)
                iteration += 1
                self.status["iteration"] = iteration

                return_dict = self._train_step(batch)

                self._train_tracker.add(len(batch))
                utils.update_metrics(return_dict, batch, self.train_metrics)

                self._fire_event(Event.Iteration, True)
            self._fire_event(Event.Epoch, True)

    def _validate_loop(self, iterator: DataIterator) -> None:
        r"""Run the validation loop given the data iterator.

        Args:
            iterator: The iterator over the validation data.
        """
        for batch in iterator:
            self._fire_event(Event.ValidationIteration, False)
            return_dict = self._validate_step(batch)

            # Update metrics.
            utils.update_metrics(return_dict, batch, self.valid_metrics)

            self._fire_event(Event.ValidationIteration, True)

    def _test_loop(self, iterator: DataIterator) -> None:
        r"""Run the entire testing loop given the data iterator.

        Args:
            iterator: The iterator over the test data.
        """
        for batch in iterator:
            self._fire_event(Event.TestingIteration, False)
            return_dict = self._test_step(batch)

            self._test_tracker.add(len(batch))
            utils.update_metrics(
                return_dict, batch, self.test_metrics)

            self._fire_event(Event.TestingIteration, True)

    def _validate(self) -> None:
        # TODO: Pause training tracker during validation.

        if self.valid_data is None:
            raise ValueError("Validation data not specified.")

        self._fire_event(Event.Validation, False)

        # Initialize metrics.
        for metric in self.valid_metrics.values():
            metric.reset()

        if self.validate_mode == "eval":
            iterator = DataIterator(self.valid_data, self.batching_strategy)
        else:
            iterator = DataIterator(self.valid_data)

        try:
            data_size: Optional[int] = len(self.valid_data)
        except TypeError:
            data_size = None
        self._valid_tracker.reset(data_size)

        model_mode = self.model.training
        self.model.train(self.test_mode == "train")

        prev_split = self.status["split"]
        self.status["split"] = "valid"

        with torch.no_grad():
            self._validate_loop(iterator)

        self._fire_event(Event.Validation, True)

        self.status["split"] = prev_split
        self.model.train(model_mode)
