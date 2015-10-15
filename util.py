# -*- coding: utf-8 -*-

"""Utilities for reformatting and analyzing spiking data"""
import numpy as np
from copy import deepcopy
from itertools import product


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