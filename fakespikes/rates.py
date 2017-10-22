"""Generate different kinds of driving rates, e.g. constant, oscillation,
and 'naturalistic'.
"""
from __future__ import division
import numpy as np
from scipy import signal


def renormalize(X, x_range, y_range):
    x1, x2 = x_range
    y1, y2 = y_range

    return (((X - x1) * (y2 - y1)) / (x2 - x1)) + y1


def create_times(t, dt):
    n_steps = int(t * (1.0 / dt))
    times = np.linspace(0, t, n_steps)

    return times


def osc(times, a, f, phase=0):
    """Sinusoidal oscillation"""

    rates = a + (a / 2.0) * np.sin(times * f * 2 * np.pi + phase)
    rates[rates < 0] = 0  # Rates must be positive

    return rates


def osc2(times, a, f, min_a=12, phase=0):
    """Continous bursting-type oscillation"""

    # Osc
    rates = np.cos((2 * np.pi * f * times) + phase)

    # Truncate negative half as a crude sim of spike bursts
    rates[rates < 0] = 0

    return renormalize(rates, (0, 1), (min_a, a))


def bursts(times, a, f, n_bursts=2, min_a=12, phase=0, offset=0, random=False):
    """Short bursts of oscillation"""

    if (not n_bursts) or (n_bursts is None):
        max_shift = int(1 / f / (times[1] - times[0]))
        rates = osc2(times, a, f, min_a, phase)
    else:
        if n_bursts < 1:
            raise ValueError("Must be at least 1 bursts.")

        # Is offset a range?
        try:
            if len(offset) == 2:
                if random:
                    raise ValueError(
                        "If an offset range is given random"
                        " must be set to False. There"
                        " is an unpredictable interaction between them.")

                offset = np.random.uniform(offset[0], offset[1])
            else:
                raise ValueError("offset must be a number or a range")
        except TypeError:
            pass

        # Break up times
        burst_l = 1 / f
        m = np.logical_and(times >= offset,
                           times < (offset + (burst_l * n_bursts)))

        max_shift = int(m.sum())

        # Build bursts
        burst = osc2(times[times <= burst_l], a, f, phase=phase)
        bursts = []
        for _ in range(n_bursts):
            bursts.extend(burst)

        # Add busts to a constant background
        rates = constant(times, a)
        rates[m] = bursts[0:m.sum()]

    if random:
        shift = int(np.random.uniform(0, max_shift))
        rates = np.roll(rates, shift)

    return rates


def square_pulse(times, a, t, w, dt, min_a=12):
    wl = int(np.round(w / dt))

    pulse = np.zeros_like(times)
    loc = (np.abs(times - t)).argmin()

    pulse[loc:loc + wl] = 1
    pulse *= a

    pulse[pulse < min_a] = min_a

    return pulse


def noisy_square_pulse(times, a, t, w, dt, sigma, min_a=12, seed=None):
    prng = np.random.RandomState(seed)

    pulse = square_pulse(times, a, t, w, dt, min_a)
    pulse += prng.normal(0, sigma, size=pulse.size)

    return pulse


def boxcar(times, a, f, w, dt, offset=0):
    """Periodic boxcars
    
    Parameters
    ----------
    times : array
        Time series (seconds)
    a : numeric
        The rate amplitude (Hz)
    f : numeric
        Boxcar frequency (Hz)
    w : numeric
        Boxcar length (seconds)
    dt : numeric
        Sampling rate
    offset : numeric
        Number of seconds to roll the phase of the series (seconds)
    """
    t = times.max()
    n_cycle = int(np.ceil(f * (t - dt)))

    # Create dirac pulses then
    l = int(np.ceil((1. / f) / dt))
    pulse = [1] + [0] * l
    pulses = []
    for _ in range(n_cycle):
        pulses.extend(pulse)

    # change to boxcars
    pulses = np.asarray(pulses, dtype=np.float)
    boxcars = pulses.copy()

    wl = int(np.round(w / dt))
    for i, p in enumerate(pulses):
        if np.isclose(p, 1):
            boxcars[i:i + wl] = 1.0

    # Scale height
    boxcars *= a

    # Change offest, if needed
    boxcars = np.roll(boxcars, int(np.round(offset / dt)))

    return boxcars[0:times.shape[0]]


def random_boxcars(times, a, n, min_w, dt, prng=None):
    if prng is None:
        prng = np.random.RandomState()

    imax = times.shape[0]

    wl = int(np.round(min_w / dt))
    box = np.ones(wl)

    idx = np.arange(0, times.shape[0], wl)
    prng.shuffle(idx)
    idx = idx[0:n]

    boxcars = np.zeros_like(times)
    for i in idx:
        boxcars[i:i + wl] = box

    boxcars *= a

    return boxcars


def inhibitory(times, a0, a, f, dt, tau_rise=9e-4, tau_decay=20e-3):
    inhib = np.zeros_like(times)

    # Locate about where the pulses of 
    # spikes will go, at f
    t = times.max()
    wl = 1 / float(f)
    n_pulses = int(t * f)
    pulses = []
    t_p = 0
    for _ in range(n_pulses):
        t_p += wl
        loc = (np.abs(times - t_p)).argmin()
        inhib[loc] += 1

    # Make the kernel, 'g'
    t0 = np.linspace(0, tau_decay * 10, (tau_decay * 10) / dt)
    g = (np.exp(-(t0 / tau_rise)) + np.exp(-(t0 / tau_decay)))

    # Convolve, norm, scale by a and subtract inhib from a constant
    # rate of a
    inhib = np.convolve(inhib, g)[0:inhib.shape[0]]
    inhib /= inhib.max()
    inhib *= a

    rates = (np.ones_like(times) * a0) - inhib
    rates[rates < 0] = 0  # Rates must be positive

    return rates


def excitatory(times, a0, a, f, dt, tau_rise=9e-4, tau_decay=20e-3):
    exc = np.zeros_like(times)

    # Locate about where the pulses of 
    # spikes will go, at f
    t = times.max()
    wl = 1 / float(f)
    n_pulses = int(t * f)
    pulses = []
    t_p = 0
    for _ in range(n_pulses):
        t_p += wl
        loc = (np.abs(times - t_p)).argmin()
        exc[loc] += 1

    # Make the kernel, 'g'
    t0 = np.linspace(0, tau_decay * 10, (tau_decay * 10) / dt)
    g = (np.exp(-(t0 / tau_rise)) + np.exp(-(t0 / tau_decay)))

    # Convolve, norm, scale by a and add exc from a constant
    # rate of a
    exc = np.convolve(exc, g)[0:exc.shape[0]]
    exc /= exc.max()
    exc *= a

    rates = (np.ones_like(times) * a0) + exc
    rates[rates < 0] = 0  # Rates must be positive

    return rates


def stim(times, d, scale, seed=None, min_rate=6):
    """Naturalistic bias (via diffusion model)"""

    normal = np.random.normal
    if seed is not None:
        prng = np.random.RandomState(seed)
        normal = prng.normal

    rates = [
        d,
    ]
    for t in times[1:]:
        d += normal(0, scale)
        rates.append(d)

    rates = np.array(rates)
    rates[rates <= min_rate] = min_rate  # Rates must be positive

    return rates


def constant(times, d):
    """Constant drive, d"""

    return np.repeat(float(d), len(times))


def noisy_constant(times, d, sigma):
    """Constant drive, d, with white noise."""

    rates = constant(times, d)
    rates += np.random.normal(0, sigma, size=times.shape[0])

    return rates
