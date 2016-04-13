# -*- coding: utf-8 -*-
"""Utilities for reformatting and analyzing spiking data"""
import numpy as np
from copy import deepcopy
from itertools import product
from scipy import signal
from scipy.stats.mstats import zscore
from scipy.signal import medfilt
from scipy.signal import resample


def to_spikes(ns, ts, T, N, dt):
    if not np.allclose(T / dt, int(T / dt)):
        raise ValueError("T is not evenly divsible by dt")

    n_steps = int(T * (1.0 / dt))
    times = np.linspace(0, T, n_steps)
    spikes = np.zeros((n_steps, N))
    for i, t in enumerate(ts):
        n = ns[i]
        idx = (np.abs(times - t)).argmin()  # find closest
        spikes[idx, n] += 1

    return spikes


def to_spiketimes(times, spikes):
    """Convert spikes to two 1d arrays"""

    n_steps = len(times)
    n = spikes.shape[1]

    ns, ts = [], []
    for i in range(n_steps):
        for j in range(n):
            if spikes[i, j] == 1:
                ns.append(j)  # save neuron and
                ts.append(times[i])  # look up dt time

    return np.array(ns), np.array(ts)


def to_spikedict(ns, ts):
    """Convert from seperate time and neuron
    arrays to a neuron-keyed dict"""
    d_sp = {}
    for n, t in zip(ns, ts):
        try:
            d_sp[n].append(t)
        except KeyError:
            d_sp[n] = [t, ]

    for k in d_sp.keys():
        d_sp[k] = np.array(d_sp[k])

    return d_sp


def spikedict_to(d_sp):
    """"Undoes `to_spikedict`"""

    ts, ns = [], []
    for n, ts_n in d_sp.iteritems():
        ns.extend([n] * len(ts_n))
        ts.extend(list(ts_n))

    return np.array(ns), np.array(ts)


def ts_sort(ns, ts):
    """Sort by ts"""
    ts = np.array(ts)
    ns = np.array(ns)
    idx = ts.argsort()

    return ns[idx], ts[idx]


def bin_times(ts, t_range, dt):
    """ts into a grid of dt sized bins"""

    if len(t_range) != 2:
        raise ValueError("t_range must contain two elements")
    if t_range[0] > t_range[1]:
        raise ValueError("t_range[0] must be less then t_range[1]")

    n_sample = int((t_range[1] - t_range[0]) * (1.0 / dt))

    bins = np.linspace(t_range[0], t_range[1], n_sample)
    binned, _ = np.histogram(ts[1:], bins=bins)

    return bins[1:], binned


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b.

    Note: a and b are two sequences
    """

    a = list(a)
    b = list(b)
    n, m = len(a), len(b)

    # Make sure n <= m, to use O(min(n,m)) space
    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[-1]


def kappa(ns1, ts1, ns2, ts2, t_range, dt):
    """Measure Bruno's Kappa correlation[0].

    [0]: Wang, X.-J. & Buzsaki, G., 1996. Gamma Oscillation by Synaptic
    Inhibition in a Hippocampal Interneuronal Network Model. J. Neurosci.,
    16(20), pp.6402–6413.
    """

    d_1 = to_spikedict(ns1, ts1)
    d_2 = to_spikedict(ns2, ts2)

    pairs = product(list(np.unique(ns1)), list(np.unique(ns2)))
    corrs = []
    for n1, n2 in pairs:
        _, b1 = bin_times(d_1[n1], t_range, dt)
        _, b2 = bin_times(d_2[n2], t_range, dt)
        b1[b1 > 1] = 1
        b2[b2 > 1] = 1
        corrs.append(_kappa(b1, b2))

    return np.nanmean(corrs)


def _kappa(bini, binj):
    return np.sum(bini * binj) / np.sqrt(np.sum(bini) * np.sum(binj))


def fano(ns, ts):
    """Calculate isi Fano"""

    d_sp = isi(ns, ts)
    d_fano = {}
    for n, v in d_sp.items():
        d_fano[n] = v.std() ** 2 / v.mean()

    return d_fano


def isi(ns, ts):
    """Return ISIs, in a neuron-keyed dict"""

    d_sp = to_spikedict(ns, ts)

    d_isi = {}
    for k, v in d_sp.items():
        tlast = 0

        intervals = []
        for t in v:
            intervals.append(t - tlast)
            tlast = deepcopy(t)

        d_isi[k] = np.array(intervals)

    return d_isi


def detect_coincidences(ns, ts, tol):
    ccs = np.zeros_like(ts)

    for i, t in enumerate(ts):
        cc = np.isclose(t, ts, atol=tol)
        ccs[i] += (cc.sum() - 1)  # Will always match self

    return ccs


def increase_coincidences(ns, ts, k, p, N, prng=None):
    if prng == None:
        prng = np.random.RandomState()
    
    ts_cc = ts.copy()
    moved = []
    for i, t in enumerate(ts):
        if i in moved:
            continue

        if p <= prng.rand():
            # k_p = prng.randint(1, k + 1)
            k_p = k
            for j in range(1, k_p + 1):
                try:
                    loc = i + j
                    ts_cc[loc] = t
                    moved.append(loc)
                except IndexError:
                    pass

    return ns, ts_cc


def dendritic_lfp(ns, ts, N, T, tau_rise=0.00009, tau_decay=5e-3,
                         dt=0.001, norm=True):
    """Simulate LFP by convloving spikes with a double exponential
    kernel

    Parameters
    ----------
    ns : array-list (1d)
        Neuron codes (integers)
    ts : array-list (1d, seconds)
        Spikes times 
    tau_rise : numeric (default: 0.00009)
        The rise time of the synapse
    tau_decay : numeric (default: 0.0015)
        The decay time of the synapse
    dt : numeric (default: 0.001)
        ??

    Note: Assumes spikes is 1 or 2d, and *column
    oriented*
    """

    spikes = to_spikes(ns, ts, T, N, dt)

    if spikes.ndim > 2:
        raise ValueError("spikes must be 1 of 2d")
    if tau_rise < 0:
        raise ValueError("tau_rise must be > 0")
    if tau_decay < 0:
        raise ValueError("tau_decay must be > 0")
    if dt < 0:
        raise ValueError("dt must be > 0")

    # Enforce col orientation if 1d
    if spikes.ndim == 1:
        spikes = spikes[:, np.newaxis]

    # 10 x tau_decay (10 half lives) should be enough to span the
    # interesting parts of g, thei double exp synaptic
    # We want 10*tau but we have to resample to dt time first
    n_syn_samples = ((tau_decay * 10) / dt)
    t0 = np.linspace(0, tau_decay * 10, n_syn_samples)

    # Define the double exp
    gmax = 1
    g = gmax * (np.exp(-(t0 / tau_rise)) + np.exp(-(t0 / tau_decay)))

    # make LFP
    spsum = spikes.astype(np.float).sum(1)
    # spsum /= spsum.max()

    lfps = np.convolve(spsum, g)[0:spikes.shape[0]]

    if norm:
        lfps = zscore(lfps)

    return lfps


def soma_lfp(ns, ts, N, T, tau=0.002, dt=.001, norm=True):
    """Simulate LFP (1d) bu convlution with an 'alpha' kernel.

    Parameters
    ----------

    ns : array-list (1d)
        Neuron codes (integers)
    ts : array-list (1d, seconds)
        Spikes times 
    tau : numeric (default: 0.001)
        The alpha estimate time constant
    dt : numeric (default: 0.001, seconds)
        Step time 
    """
    spikes = to_spikes(ns, ts, T, N, dt)

    if spikes.ndim > 2:
        raise ValueError("spikes must be 1 of 2d")
    if tau < 0:
        raise ValueError("tau must be > 0")
    if dt < 0:
        raise ValueError("dt must be > 0")

    # Enforce col orientation if 1d
    if spikes.ndim == 1:
        spikes = spikes[:, np.newaxis]

    # 10 x tau (10 half lives) should be enough to span the
    # interesting parts of g, the alpha function we are
    # using to convert broadband firing to LFP
    # a technique we are borrowing from:
    #
    # http://www.ncbi.nlm.nih.gov/pubmed/20463210
    #
    # then abusing a bit (too much?).
    #
    # We want 10*tau but we have to resample to dt time first
    n_alpha_samples = ((tau*10)/dt)
    t0 = np.linspace(0, tau*10, n_alpha_samples)

    # Define the alpha (g notation borrow from BV's initial code)
    gmax = 0.1
    g = gmax * (t0 / tau) * np.exp(-(t0 - tau) / tau)

    # make LFP
    spsum = spikes.astype(np.float).sum(1)
    spsum /= spsum.max()

    lfps = np.convolve(spsum, g)[0:spikes.shape[0]]

    if norm:
        lfps = zscore(lfps)

    return lfps

