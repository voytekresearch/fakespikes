# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import RandomState


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
        if (refractory % dt) != 0:
            raise ValueError("refractory must be integer multiple of dt")

        self.n = n
        self.refractory = refractory

        # Timing
        self.dt = dt
        self.t = t
        self.n_steps = int(self.t * (1.0 / self.dt))
        self.times = np.linspace(0, self.t, self.n_steps)
        self.private_stdev = private_stdev

        if refractory % self.dt != 0:
            raise ValueError("refractory must be a integer multiple of dt")
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
            mask = self.unifs[:,j] <= ((rates + biases[j]) * self.dt)
            spikes[mask,j] = 1

        return self._refractory(spikes)
