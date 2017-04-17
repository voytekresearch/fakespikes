# -*- coding: utf-8 -*-
"""Utilities for reformatting and analyzing spiking data"""
import numpy as np
from copy import deepcopy
from itertools import product
from scipy import signal
from scipy.stats.mstats import zscore
from scipy.signal import medfilt
from scipy.signal import resample
from scipy.signal import welch, gaussian
from scipy.stats import entropy


def create_times(t, dt):
    n_steps = int(t * (1.0 / dt))
    times = np.linspace(0, t, n_steps)

    return times


def create_psd(lfp, inrate, outrate=1024):
    """Calculate PSD from LFP/EEG data."""
    lfp = np.array(lfp)

    if inrate != outrate:
        lfp = signal.resample(lfp, int(lfp.shape[0] * outrate / inrate))

    # Calculate PSD
    return signal.welch(
        lfp,
        fs=outrate,
        window='hanning',
        nperseg=outrate,
        noverlap=outrate / 2.0,
        nfft=None,
        detrend='linear',
        return_onesided=True,
        scaling='density')


def select_n(sel, ns, ts):
    """Select some neurons.

    Params
    ------
    sel : list 
        Which neurons to select
    ts : array-like (1d)
        Spike times
    ns : array-like (1d)
        Neurons  
    """
    if ns.shape != ts.shape:
        raise ValueError("ns and ts must be the same shape")

    m = np.zeros_like(ts, dtype=np.bool)

    for n in sel:
        m = np.logical_or(m, n == ns)

    return ns[m], ts[m]


def to_spikes(ns, ts, T, N, dt):
    """Convert spike times to a grid and binary representation"""

    if not np.allclose(T / dt, int(np.round(T / dt))):
        raise ValueError("T is not evenly divsible by dt")

    n_steps = int(T * (1.0 / dt))
    times = np.linspace(0, T, n_steps)
    spikes = np.zeros((n_steps, N))

    for i, t in enumerate(ts):
        n = ns[i]
        idx = (np.abs(times - t)).argmin()  # find closest
        spikes[idx, n - 1] += 1

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
            d_sp[n] = [
                t,
            ]

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


def coincidence_code(ts, ns, tol):
    """Define a spike-time coincidence code

    Params
    ------
    ts : array-like (1d)
        Spike times
    ns : array-like (1d)
        Neurons  
    tol : numeric
        How close two spikes must be to be coincident
    """

    # The encoded sequence
    encoded = []
    ts_e = []

    for i, t in enumerate(ts):
        # Find which neurons fired in coincidence
        # and count them. The count is the code.
        m = np.isclose(t, ts, atol=tol)
        n_set = np.sum(m)

        encoded.append(n_set)
        ts_e.append(t)

    return np.asarray(encoded), np.asarray(ts_e)


def spike_time_code(ts, scale=1000, decimals=3):
    """Spike time code 
    
    Note: ts MUST be in seconds

   Params
    ------
    ts : array-like (1d)
        Spike times
    ns : array-like (1d)
        Neurons
    scale : numeric
        Amount to scale `ts` by before int conversion
    """

    scaled = np.round(ts, decimals) * scale
    encoded = scaled.astype(np.int)

    return encoded


def spike_window_code(ts, ns, dt=1e-3):
    """Define a spike-time window code

    Params
    ------
    ts : array-like (1d)
        Spike times
    ns : array-like (1d)
        Neurons  
    dt : numeric
        Window size (seconds)
    """

    # The encoded sequences
    encoded = []
    ts_e = []

    # The encoding machinery
    encoding = {}
    master_code = 0
    for t in ts:
        # Find which neurons fired in coincidence
        # and make them a set
        m = np.isclose(t, ts, atol=dt)
        n_set = frozenset(ts[m])

        # If this set isn't known yet use the
        # master code to encode it
        try:
            encoding[n_set]
        except KeyError:
            encoding[n_set] = master_code
            master_code += 1

        # Finally do the encode
        encoded.append(encoding[n_set])
        ts_e.append(t)

    return np.asarray(encoded), np.asarray(ts_e), encoding


def rate_code(ts, t_range, dt, k=1):
    """Define a rate code

    Params
    ------
    ts : array-like
        spike times
    t_range : 2-tuple
        (Min, Max) values of ts
    dt : numeric
        Window size which which to bin
    k : numeric 
        The effective resolution of the rate code (Hz)

    Notes
    -----
    While the absolute value/units of t, t_range 
    and dt do not matter, their values must be self
    consistent.
    """

    # resample, discretize and encode
    # convert magnitudes to order of apperance
    # e.g. [1, 2, 5, 1, 3, 4] becomes
    #      [1, 2, 3, 1, 4, 5]

    # 1. bin times
    t_bins, binned = bin_times(ts, t_range, dt)

    # 2. norm rate ranage
    max_r = binned.max()
    n_d = np.int(np.ceil(max_r / k))
    digitized = np.digitize(binned, np.linspace(0, max_r, n_d))

    # 3. Encode by order of apperance
    # Define encodes by order of appearance in digitized
    # This eocnding makes the order of rate changes matter
    # not the overall amplitude or exact binning details
    # of the rate matter

    # The encoding machinery
    master_code = 0
    encoding = {}

    # The encoded sequence
    encoded = []

    for d in digitized:
        # Init is this the first time seeing d
        try:
            encoding[d]
        except KeyError:
            encoding[d] = master_code
            master_code += 1

        encoded.append(encoding[d])

    # return np.asarray(encoded), t_bins
    return np.asarray(encoded), t_bins, encoding


def spike_triggered_average(ts, ns, trace, t_range, dt, srate):
    """Spike triggered average

    Return the spike triggered average or trace, in a window
    of width dt.

    Params
    ------
    ts : array
        Spike times
    trace : array
        The data to average
    t_range : 2-tuple
        The (min, max) values to trace
    dt : numeric
        The window size
    srate : numeric
        The sampling rate of trace
    """
    n_bins = int(np.ceil((2 * (dt * srate))))
    bins = np.linspace(-dt, dt, n_bins)

    n_steps = int(np.ceil(srate * t_range[1]))
    times = np.linspace(t_range[0], t_range[1], n_steps)

    sta = np.zeros(n_bins)

    # Sanity: check for empty times or ns 
    # and return 0s in the sta if needed
    if (ts.size == 0) or (ns.size == 0):
        return sta, bins

    for t, n in zip(ts, ns):
        # Prevent over/underflow
        if t < dt:
            continue
        if t > (t_range[1] - dt):
            continue

        # Define the window and sum it
        m = np.logical_and(times >= (t - dt), times <= (t + dt))
        sta += trace[n, m]  # Avg over neurons at each t

    sta /= ts.size  # divide the sum by n -> the mean.

    return sta, bins


def _countent(p):
    mp = np.ma.array(p, copy=False, mask=np.isclose(p, 0.0), fill_value=0.0)
    return (p * np.log(mp)).sum(axis=0)


def mi(X, Y):
    """Discrete mutual information (no bias correction)

    Note: Only supports 1d inputs, and integer values.
    """

    if (not np.issubdtype(X.dtype, np.int)) or (
            not np.issubdtype(Y.dtype, int)):
        raise ValueError, "Input should have integer dtype"
    if (len(X.shape) > 1) or (len(Y.shape) > 1):
        raise ValueError, "Only 1D inputs supported"

    # Sizes
    Nx = len(set(X))
    Ny = len(set(Y))

    Nt = Nx + Ny
    logNt = np.log(Nt)
    Ntr = np.float(Nt)

    # Init p(.)
    Px = np.zeros(Nx)
    Py = np.zeros(Ny)
    Pxy = np.zeros(Nx * Ny)

    # Est p(.)
    for i in range(Nt):
        Px[X[i]] += 1
        Py[Y[i]] += 1

        idx = (X[i] * Ny) + Y[i]
        Pxy[idx] += 1

    # Est H
    HX = logNt - (_countent(Px) / Ntr)
    HY = logNt - (_countent(Py) / Ntr)
    HXY = logNt - (_countent(Pxy) / Ntr)

    # MI
    I = HX + HY - HXY
    I = I / np.log(2.0)

    return I


def kl_divergence(a, b):
    """Calculate the K-L divergence between a and b
    
    Note: a and b must be two sequences of integers
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Find the total set of symbols
    a_set = set(a)
    b_set = set(b)
    ab_set = a_set.union(b_set)

    # Create a lookup table for each symbol in p_a/p_b
    lookup = {}
    for i, x in enumerate(ab_set):
        lookup[x] = i

    # Calculate event probabilities for and then b
    # To prevent nan/division errors every event
    # gets at least a 1 count.
    p_a = np.ones(len(ab_set))
    for x in a:
        p_a[lookup[x]] += 1

    p_b = np.ones(len(ab_set))
    for x in b:
        p_b[lookup[x]] += 1

    # Norm counts into probabilities
    p_a /= a.size
    p_b /= b.size

    return entropy(p_a, p_b, base=2)


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
        d_fano[n] = v.std()**2 / v.mean()

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


def spiketimes_to_coincidences(ns, ts, tol):
    t_cc = []
    n_cc = []

    for i, t in enumerate(ts):
        cc = np.isclose(t, ts, atol=tol)
        t_cc.append()
        ccs[i] += (cc.sum() - 1)  # Will always match self


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


def precision(ns, ts, ns_ref, ts_ref, combine=True):
    """Analyze spike time precision (jitter)
    
    Parameters
    ----------
    ns : array-list (1d)
        Neuron codes 
    ts : array-list (1d, seconds)
        Spikes times 
    ns_ref : array-list (1d)
        Neuron codes for the reference train
    ts_ref : array-list (1d, seconds)
        Spikes times for the reference train
    """

    prec = []
    ns_prec = []

    if combine:
        ns = np.zeros_like(ns)
        ns_ref = np.zeros_like(ns_ref)

    # isolate units, and reformat
    ref = to_spikedict(ns_ref, ts_ref)
    target = to_spikedict(ns, ts)

    # analyze precision
    for n, r in ref.iteritems():
        x = target[n]
        minl = min(len(r), len(x))
        diffs = np.abs([r[i] - x[i] for i in range(minl)])

        prec.append(np.mean(diffs))
        ns_prec.append(n)

    # If were are combining return scalars
    # not sequences
    if combine:
        prec = prec[0]
        ns_prec = ns_prec[0]

    return ns_prec, prec


def seperation(ns_1, ts_1, ns_2, ts_2, dt, T=None):
    """Estimate the distance between two populations, 
    using the Gaussian smoothed PSTH.

    Parameters
    ----------
    ns_1 : array-list (1d)
        Neuron codes 
    ts_1 : array-list (1d, seconds)
        Spikes times
    ns_2 : array-list (1d)
        Neuron codes 
    ts_2 : array-list (1d, seconds)
        Spikes times
    dt : numeric
        Sampling resolution
    """

    # calculate the PSTH
    if T is None:
        T = max(ts_1.max(), ts_2.max())
    N = max(ns_1.max(), ns_2.max())

    psth1 = to_spikes(ns_1, ts_1, T, N, dt).sum(1)
    psth2 = to_spikes(ns_2, ts_2, T, N, dt).sum(1)

    # convolve, sigma 5 ms
    sigma = 5e-3
    lwin = int((3 * sigma) / dt)
    g = gaussian(lwin, sigma / dt)

    # calculate the seperation
    psth1 = np.convolve(psth1, g)[0:psth1.shape[0]]
    psth2 = np.convolve(psth2, g)[0:psth2.shape[0]]
    sd1 = np.std(psth1)
    sd2 = np.std(psth2)
    sd = np.sqrt((sd1**2) + (sd2**2))

    return np.abs(psth1 - psth2) / sd, psth1, psth2


def dendritic_kernel(tau_rise, tau_decay, dt, gmax=1):
    """Simulate a PSP, with a double exponential."""

    # 10 half lives is enough to capture the response...
    n_syn_samples = ((tau_decay * 10) / dt)
    t0 = np.linspace(0, tau_decay * 10, n_syn_samples)

    tpeak = tau_decay * tau_rise / (tau_decay - tau_rise) * np.log(tau_decay /
                                                                   tau_rise)
    normf = 1 / (-np.exp(-tpeak / tau_rise) + np.exp(-tpeak / tau_decay))
    g = (-np.exp(-t0 / tau_rise) + np.exp(-t0 / tau_decay)) * normf
    g *= gmax

    return g


def dendritic_lfp(ns,
                  ts,
                  N,
                  T,
                  tau_rise=0.00009,
                  tau_decay=5e-3,
                  gmax=1,
                  dt=0.001,
                  norm=True):
    """Simulate LFP by convloving spikes with a double exponential
    kernel

    Parameters
    ----------
    ns : array-list (1d)
        Neuron codes (integers)
    ts : array-list (1d, seconds)
        Spikes times 
    N : numeric
        Total number of neurons
    T : number
        Total length of firing
    tau_rise : numeric (default: 0.00009)
        The rise time of the synapse
    tau_decay : numeric (default: 0.0015)
        The decay time of the synapse
    dt : numeric (default: 0.001)
        Time resolution (in seconds)
    gmax : numeric
        PSP max height

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

    # make LFP
    g = dendritic_kernel(tau_rise, tau_decay, dt, gmax)
    spsum = spikes.astype(np.float).sum(1)
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
    n_alpha_samples = ((tau * 10) / dt)
    t0 = np.linspace(0, tau * 10, n_alpha_samples)

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
