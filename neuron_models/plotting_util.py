from pylab import *
from brian import *
from util import *

def raster_plot_subset(spikes, inds, axis=None, **plotoptions):
    '''
    display a raster plot of neurons indexed by inds
    
    spikes: dictionary of arrays containing spike times
            spikes[neuron_index] = array([t1,t2,...tn])
            
    spikes = SpikeMonitor.spiketimes()
    '''
    spiketimes_subset = []
    neuron_numbers = []
    for i in inds:
        spiketimes_subset.append(spikes[i])
        neuron_numbers.append(i*ones_like(spikes[i]))
    times = (hstack(spiketimes_subset)/ms).astype(int)
    if axis is None:
        plot(times,hstack(neuron_numbers),'.',mew=0,**plotoptions)
        ylim([min(inds),max(inds)])
    else:
        axis.plot(times,hstack(neuron_numbers),'.',mew=0,**plotoptions)
        axis.set_ylim(ymax=max(inds))
        axis.set_ylabel('Neuron number')
    
def plot_population_firing_rate(spikes, n_bins, tau=25., ax=None):
    '''
    plot the neuron average firing rate for the population
    response
    '''
    spike_bins = zeros(n_bins)
    for n,n_spike_times in spikes.iteritems():
        spike_bins[(n_spike_times/ms).astype(int)] += 1
    spike_bins /= len(spikes) # per neuron average
    window = flipud(exp(arange(200)/tau))
    window /= window.sum()
    mean_firing_rate = convolve(spike_bins, window)[:spike_bins.shape[0]]*1000
    #spike_bins_ = pad(spike_bins,(99,99),'constant',constant_values=(mean(spike_bins[:100]),mean(spike_bins[-100:])))
    #mean_firing_rate = convolve(spike_bins_,10.*ones(100),'same')[99:-99]
    if ax is None:
        plot(mean_firing_rate)
    else:
        ax.plot(mean_firing_rate,'k',linewidth=2)
        ax.set_ylabel('Average Neuron Firing Rate Hz')
        
def plot_raster_firingrate_overlay(spikes, n_bins, inds, ax1):
    '''
    raster plot of spikes in monitor, only of neurons indexed by inds
    overlayed by the firing rate curve
    '''
    raster_plot_subset(spikes,inds,axis=ax1,alpha=.4)
    ax2 = ax1.twinx()
    plot_population_firing_rate(spikes,n_bins,ax=ax2)
    
    
def plot_raster_across_trials_for(neuron_ind, trials_spikes, ax=None, **plotoptions):
    neuron_spiketimes, trial_indices = [],[]
    i = 0
    for trial in trials_spikes:
        neuron_spiketimes.append(trial[neuron_ind])
        trial_indices.append(i*ones_like(trial[neuron_ind]))
        i += 1
    neuron_spiketimes = (hstack(neuron_spiketimes)*1000).astype(int)
    if ax is None:
        plot(neuron_spiketimes,hstack(trial_indices),'.',mew=0,**plotoptions)
    else:
        ax.plot(neuron_spiketimes,hstack(trial_indices),'.',mew=0,**plotoptions)
        ax.set_ylabel('Trial number')
        
def plot_raster_across_trials_with_firing_rate(neuron_ind, trials_spikes, ax1, time_bins):
    plot_raster_across_trials_for(neuron_ind,trials_spikes, ax=ax1, alpha=.4)
    ax2 = ax1.twinx()
    ax2.plot(compute_firing_rate_across_trials_for(neuron_ind, trials_spikes, time_bins),'k')
    ax2.set_ylabel('Average firing rate (Hz)')

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
def plot_ISI_histogram(ISI_monitor, spike_monitor, rate_monitor, xy, xytext, color='w'):
    hist_plot(ISI_monitor, newfigure=False, color=color)
    mew, std = isi_mean_and_std(spike_monitor)
    s = 'rate = %0.1f Hz\nCV = %0.2f' % (mean(rate_monitor.rate),std/mew)
    annotate(s,xy=xy,xytext=xytext,xycoords='data',fontsize=24)
    tick_params(labelsize=20)
    xlabel('ISI (ms)', fontsize=20)
    ylabel('Count', fontsize=20)
    
def plot_spike_correlogram(T1, T2, width=20 * ms, bin=1 * ms, T=None, auto_ylim=True):
    '''
    MODIFIED from brian.tools.statistics.correlogram

    T1,T2 are ordered arrays of spike times.

    Returns a cross-correlogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is number of coincidences in each bin.

    auto_ylim automatically sets the ylim to be 1.2 times the second greatest value in the correlogram.

    N.B.: units are discarded.
    '''
    if (T1==[]) or (T2==[]): # empty spike train
        return NaN
    # Remove units
    width = float(width)
    T1 = array(T1)
    T2 = array(T2)
    i = 0
    j = 0
    n = int(ceil(width / bin)) # Histogram length
    l = []
    for t in T1:
        while i < len(T2) and T2[i] < t - width: # other possibility use searchsorted
            i += 1
        while j < len(T2) and T2[j] < t + width:
            j += 1
        l.extend(T2[i:j] - t)
    H, _ = histogram(l, bins=arange(2 * n + 1) * bin - n * bin)
    p = subplot(111)
    p.plot(linspace(-width*1000,width*1000,H.shape[0]),H,color='k')
    p.fill_between(linspace(-width*1000,width*1000,H.shape[0]),H,color='k')
    xlim([-width*1000,width*1000])
    tick_params(labelsize=16)
    xlabel('time (ms)',fontsize=20)
    ylabel('counts',fontsize=20)
    if auto_ylim:
        ylim([0,H[:H.shape[0]/2].max()*1.2])