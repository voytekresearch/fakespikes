"""Generate different kinds of driving rates, e.g. constant, oscillation,
and 'naturalistic'.
"""
import numpy as np


def osc(times, a, f):
    """Oscillating bias term"""

    return a + (a / 2.0) * np.sin(times * f * 2 * np.pi)


def stim(times, d, scale, seed=None):
    """Naturalistic bias (via diffusion model)"""

    normal = np.random.normal
    if seed is not None:
        prng = np.random.RandomState(seed)
        normal = prng.normal

    rates = [d, ]
    for t in times[1:]:
        d += normal(0, scale)
        rates.append(d)

    rates = np.array(rates)
    rates[rates < 0] = 0  # Rates must be positive

    return rates


def constant(times, d):
    """Constant drive, d"""

    return np.repeat(d, len(times))


