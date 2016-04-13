"""Generate different kinds of driving rates, e.g. constant, oscillation,
and 'naturalistic'.
"""
import numpy as np


def osc(times, a, f):
    """Oscillating bias term"""

    rates = a + (a / 2.0) * np.sin(times * f * 2 * np.pi)
    rates[rates < 0] = 0  # Rates must be positive

    return rates 


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
        loc = (np.abs( times - t_p)).argmin()
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
        loc = (np.abs( times - t_p)).argmin()
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
