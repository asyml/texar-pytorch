"""
Model utility functions
"""


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    linear decay.
    """
    step = min(step, total_steps)

    multiplier = (1 - (step - warmup_steps) / (total_steps - warmup_steps))

    if warmup_steps > 0 and step < warmup_steps:
        warmup_percent_done = step / warmup_steps
        multiplier = warmup_percent_done

    return multiplier
