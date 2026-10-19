"""Small offline helpers for the draft WP2 factorial analysis plan."""
from math import isfinite


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in input order."""
    if not isinstance(p_values, (tuple, list)) or not p_values:
        raise ValueError('a nonempty p-value family is required')
    if any(type(p) not in (int, float) or not isfinite(p) or not 0. <= p <= 1.
           for p in p_values):
        raise ValueError('p-values must be finite numbers in [0, 1]')
    size = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda row: row[1])
    adjusted = [0.] * size
    running = 0.
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1., (size-rank)*value))
        adjusted[index] = running
    return tuple(adjusted)
