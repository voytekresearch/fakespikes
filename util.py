"""Utilities for reformatting and analyzing spiking data"""
import numpy as np
from copy import deepcopy


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
    """Convert from arrays to a neuron-keyed dict"""
    d_sp = {}
    for n, t in zip(ns, ts):
        try:
            d_sp[n].append(t)
        except KeyError:
            d_sp[n] = [t, ]

    for k in d_sp.keys():
        d_sp[k] = np.array(d_sp[k])

    return d_sp


def pop_rate(spikes, t, avg=False):
    """Calculate the Population firing rate.

    Params
    ------
    spikes : 2d array
        The spikes (on a grid)
    t : scalar
        Total simulation time (seconds)
    avg : bool
        Return the avg firing rate?
    """

    prate = spikes.sum()
    if avg:
        prate = prate.astype(np.float)
        prate /= t
        prate /= spikes.shape[1]

    return prate


def t_rate(spikes, avg=False):
    """Calculate the Population firing rate in time.

    Params
    ------
    spikes : 2d array
        The spikes (on a grid)
    avg : bool
        Return the avg firing rate?
    """
    trate = spikes.sum(1)
    if avg:
        trate = trate.astype(np.float)
        trate /= spikes.shape[1]

    return trate


def n_rate(spikes, t, avg=False):
    """Calculate the Population firing rate for each neuron.

    Params
    ------
    spikes : 2d array
        The spikes (on a grid)
    avg : bool
        Return the avg firing rate?
    """

    nrate = spikes.sum(0)
    if avg:
        nrate = nrate.astype(np.float)
        nrate /= t

    return nrate


def fano(spikes):
    """Calculate spike-count Fano"""
    return spikes.sum(0).std() ** 2 / spikes.sum(0).mean()


def isi(d_sp):
    """ISIs, in a neuron-keyed dict"""

    d_isi = {}
    for k, v in d_sp.items():
        tlast = 0

        intervals = []
        for t in v:
            intervals.append(t - tlast)
            tlast = deepcopy(t)

        d_isi[k] = np.array(intervals)

    return d_isi
