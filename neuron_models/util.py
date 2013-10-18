import numpy as np
from brian import *

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    SOURCE: http://stackoverflow.com/a/1235363
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def percent_active_cells(monitor, bin_len):
    active_bins = zeros(monitor.source.clock.t/ms)
    for n,spikes in monitor.spiketimes.iteritems():
        bins = zeros(monitor.source.clock.t/ms)
        for t in (spikes/ms).astype(int):
            bins[t-bin_len/2:t+bin_len/2] = 1
        active_bins += bins
    return active_bins/len(monitor.source)

def output_spikes_to(flat_file, spike_times):
    '''
    save spikes to disk in format:
    spike_time(int) neuron_number\n
    
    spike_times is a dictionary of arrays
    spiketimes[neuron_number] = array([t1, t2, .. tn])
    '''
    spike_tuples = []
    for ind, times in spike_times.iteritems():
        for t in (times / ms).astype(int):
            spike_tuples.append((t,ind))
    with open(flat_file,'w') as f:
        for t,i in sorted(spike_tuples):
            f.write('%s %s\n' % (t,i))

def load_spikes_from(flat_file):
    '''
    loads spikes from flat file and returns
    a dictionary of arraysc containing spike times
    d[neuron_number] = array([t1,t2,...,tn])
    '''
    with open(flat_file) as inf:
        d = {}
        for l in inf.readlines():
            t,n = l.strip().split(' ')
            try:
                d[int(n)].append(int(t)*ms)
            except KeyError:
                d[int(n)] = []
                d[int(n)].append(int(t)*ms)
        for n,s in d.iteritems():
            d[n] = np.array(s)
    return d

def compute_firing_rate_across_trials_for(neuron_ind, trials_spikes, time_bins, **plotoptions):
    spike_bins = zeros(time_bins)
    for trial in trials_spikes:
        spike_bins[(trial[neuron_ind]*1000).astype(int)] += 1
    spike_bins /= len(trials_spikes)
    spike_bins_ = pad(spike_bins,(99,99),'constant',constant_values=(mean(spike_bins[:100]),mean(spike_bins[-100:])))
    return convolve(spike_bins_,10.*ones(100),'same')[99:-99]
    
def isi_mean_and_std(monitor):
    '''
    compute the mean and variance of interspike intervals
    of a group of neurons
    '''
    isi = []
    for n_ind, times in monitor.spiketimes.iteritems():
        isi += list(diff(times)*1000)
    return mean(isi), var(isi)**.5

def extract_two_spike_trace(V_trace, spiketimes, window, dt):
    '''
    This function extracts the voltage trace between the first two spikes plus
    a small window on either side.

    V_trace: array, voltage trace of the neuron
    spiketimes: array, list of spike times
    window: time length (in units of seconds) of trace before and after first and second spikes, 
    respectively, to extract.
    dt: the time step of the trace (in units of seconds)
    '''
    spike_inds = find(V_trace==0)
    i1, i2 = spike_inds[0], spike_inds[1]
    window_len = int(window/dt)
    return V_trace[i1-window_len:i2+window_len]