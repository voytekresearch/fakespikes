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
