"""A quick demo of oscillation, stimulation, and background noise."""
import numpy as np
from fakespikes import neurons, util, rates
import seaborn as sns
import matplotlib.pyplot as plt
sns.__file__  # pylint
plt.ion()

# -- USER SETTINGS -----------------------------------------------------------
seed = 42
n = 50  # neuron number
t = 5  # run 10 seconds

Istim = 2  # Avg rate of 'natural' stimulation
Sstim = 0.1 * Istim  # Avg st dev of natural firing
Iosc = 10  # Avg rate of the oscillation
f = 1  # Freq of oscillation
Iback = 10  # Avg rate of the background noise

# Timing
dt = 0.001
rate = 1 / dt

# -- SIM ---------------------------------------------------------------------
# Init spikers
nrns = neurons.Spikes(n, t, dt=dt, seed=42)
times = nrns.times  # brevity

# Create biases
osc = rates.osc(times, Iosc, f)
stim = rates.stim(times, Istim, Sstim, seed)
noise = rates.constant(times, Iback)

# Simulate spiking
spks_osc = nrns.poisson(osc)
spks_stim = nrns.poisson(stim)
spks_noise = nrns.poisson(noise)


# Reformat and plot a raster
spks = np.hstack([spks_osc, spks_stim, spks_noise])  # Stack 'em for plotting
ns, ts = util.to_spiketimes(times, spks)
plt.plot(ts, ns, 'o')

# and their summed rates
plt.figure()
plt.subplot(311)
plt.plot(times, spks_osc.sum(1), label='osc')
plt.legend()

plt.subplot(312)
plt.plot(times, spks_stim.sum(1), label='stim')
plt.legend()

plt.subplot(313)
plt.plot(times, spks_noise.sum(1), label='background')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Rate")
