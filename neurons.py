# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import RandomState
from fakespikes.util import to_spikes
from fakespikes.rates import constant


class Spikes(object):
    """Simulates statistical models of neural spiking

    Params
    ------
    n : int
        Number of neurons
    t : float
        Simulation time (seconds)
    dt : float
        Time-step (seconds)
    refractory : float
        Absolute refractory time
    seed : None, int, RandomState
        The random seed
    private_stdev : float
        Amount of stdev noise to add to each neurons tuning respose
    """
    def __init__(self, n, t, dt=0.001, refractory=0.002, seed=None,
                 private_stdev=0):

        # Ensure reproducible randomess
        self.seed = seed
        if isinstance(seed, RandomState):
            self.prng = seed
        elif self.seed is not None:
            self.prng = np.random.RandomState(seed)
        else:
            self.prng = np.random.RandomState()

        # Init constraints
        if n < 2:
            raise ValueError("n must be greater than 2")
        if dt > 0.001:
            raise ValueError("dt must be less than 0.001 seconds (1 ms)")
        if not np.allclose(refractory / dt, int(refractory / dt)):
            raise ValueError("refractory must be integer multiple of dt")

        self.n = n
        self.refractory = refractory

        # Timing
        self.dt = dt
        self.t = t
        self.n_steps = int(self.t * (1.0 / self.dt))
        self.times = np.linspace(0, self.t, self.n_steps)
        self.private_stdev = private_stdev
        self.refractory = refractory

        # Create uniform sampling distributions for each neuron
        self.unifs = np.vstack(
            [self.prng.uniform(0, 1, self.n_steps) for i in range(self.n)]
        ).transpose()

    def _constraints(self, drive):
        if drive.shape != self.times.shape:
            raise ValueError("Shape of `drive` didn't match times")
        if drive.ndim != 1:
            raise ValueError("`drive` must be 1d")

    def _refractory(self, spks):
        lw = int(self.refractory / self.dt)  # len of refractory window

        # If it spiked at t, delete spikes
        # in the refractory window
        for t in range(spks.shape[0]):
            mask = spks[t, :]
            for t_plus in range(lw):
                spks[t_plus, :][mask] = 0

        return spks

    def poisson(self, rates):
        """Simulate Poisson firing

        Params
        ------
        rates : array-like, 1d, > 0
            The firing rate
        """
        self._constraints(rates)  # does no harm to check twice

        # No bias unless private_stdev is specified
        biases = np.zeros(self.n)
        if self.private_stdev > 0:
            biases = self.prng.normal(0, self.private_stdev, size=self.n)

        # Poisson method taken from
        # http://www.cns.nyu.edu/~david/handouts/poisson.pdf
        spikes = np.zeros_like(self.unifs, np.int)
        for j in range(self.n):
            mask = self.unifs[:, j] <= ((rates + biases[j]) * self.dt)
            spikes[mask, j] = 1

        return self._refractory(spikes)


    def sync_bursts(self, a0, f, k, var=1e-3):
        """Create synchronous bursts (1 ms variance) of thalamic-ish spike

        Params
        ------
        f : numeric
            Oscillation frequency (Hz)
        k : numeric
            Number of neuron to spike at a time
        """

        if k > self.n:
            raise ValueError("k is larger than N")
        if f < 0:
            raise ValueError("f must be greater then 0")
        if k < 0:
            raise ValueError("k must be greater then 0")

        
        # Locate about where the pulses of spikes will go, at f,
        wl = 1 / float(f)
        n_pulses = int(self.t * f)
        pulses = []
        t_p = 0
        for _ in range(n_pulses):
            t_p += wl

            # Gaurd against negative ts
            if t_p > (3 * var):
                pulses.append(t_p)

        # and fill in the pulses with Gaussin distributed spikes.
        Ns = range(self.n)
        ts = []
        ns = []
        for t in pulses:
            ts += list(t + self.prng.normal(0, var, k))

            # Assign spikes to random neurons, at most
            # one spike / neuron
            self.prng.shuffle(Ns)
            ns += list(Ns)[0:k]

        ts = np.array(ts)
        ns = np.array(ns)

        # Just in case any negative time any slipped trough
        mask = ts > 0
        ts = ts[mask]
        ns = ns[mask]
        spikes = to_spikes(ns, ts, self.t, self.n, self.dt)
        
        # Create baseline firing
        base = self.poisson(constant(self.times, a0))
        
        spikes = base + spikes
        spikes[spikes > 1] = 1

        return spikes

