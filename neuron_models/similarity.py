import numba as nb
from numba.decorators import autojit
from pylab import *
from brian import ms
from scipy.ndimage.filters import convolve1d

def cluster_spike_bins(spike_times, N_GR, N_GO, T):
    '''
    computes the population average activity at each point in time
    for each granule cell cluster. (i.e. number of spikes 
    per unit time of each granule cell cluster)
    
    returns [n_bins = N_GO, T] ndarray
    '''
    spike_bins = zeros((N_GO,T))
    R = N_GR/N_GO
    for n,times in spike_times.iteritems():
        spike_bins[int(n)/R][(times/ms).astype(int)] += 1./R
    return spike_bins

def compute_population_avg_activity(bins,tau=8.3):
    weight_window = flipud(np.exp(-arange(bins.shape[1])/tau))
    c = convolve1d(fliplr(bins),weight_window,mode='constant',cval=0.,
                   axis=1,origin=bins.shape[1]/2-1)
    return fliplr(c)/tau

@autojit(arg_types=[nb.double[:,:],nb.double[:,:],nb.double[:],nb.double[:]])
def compute_correlation(spike1,spike2,norm1,norm2):
    '''
    computes the correlation between spike1 and spike2
    across all bins/sub populations
    
    spike1: [n_bins, time]
    spike2: same
    '''
    N,T = spike1.shape[0], spike1.shape[1]
    c = zeros((T,2*T))
    for t1 in range(T):
        for t2 in range(t1,t1+T):
            if 0 <= t2 < T: 
                dt2 = t2
            elif t2 < 0:
                dt2 = t2 + T
            else:
                dt2 = t2 - T
            
            r = 0.
            for i in range(N):
                r += spike1[i,t1]*spike2[i,dt2]
            
            if norm1[t1] > 0 and norm2[dt2] > 0:
                c[t1,t2-t1] = r/(norm1[t1]*norm2[dt2])
            else:
                c[t1,t2-t1] = 0
    return c

def population_spike_similarity(spikes1, spikes2, N_GR, N_GO, T, tau=8.3):
    '''
    computes a similarity measure between two population spike trains
    
    spikes1: dictionary {neuron_index:spike_times_array}.  spike times
    are assumed to be encoded in milliseconds, e.g. .001 is 1 ms.
    
    see Yamazaki and Nagao 2012 for details
    '''
    spike_bins1 = cluster_spike_bins(spikes1,N_GR,N_GO,T)
    z1 = compute_population_avg_activity(spike_bins1, tau)
    norm_z1 = (z1 ** 2).sum(axis=0) **.5
    if spikes2 is None:
        z2 = z1
        norm_z2 = norm_z1
    else:
        spike_bins2 = cluster_spike_bins(spikes2,N_GR,N_GO,T)
        z2 = compute_population_avg_activity(spike_bins2, tau)
        norm_z2 = (z2 ** 2).sum(axis=0) **.5
    sim = compute_correlation(z1,z2,norm_z1,norm_z2).mean(axis=0)
    return hstack((sim[T/2+1:T],sim[:T/2]))

def population_spike_correlation(spikes1,spikes2,N_GR, N_GO, T, tau=8.3):
    '''
    computes a similarity measure between two population spike trains
    
    spikes1: dictionary {neuron_index:spike_times_array}.  spike times
    are assumed to be encoded in milliseconds, e.g. .001 is 1 ms.
    
    if spikes2 is None, autocorrelation is computed
    
    NOTE: This replaces population_spike_similarity() and 
    compute_correlation() with a faster implementation using
    numpy
    '''
    bins1 = cluster_spike_bins(spikes1,N_GR,N_GO,T)
    activity1 = compute_population_avg_activity(bins1, tau)
    norm1 = ((activity1**2).sum(axis=0))**.5
    activity1 /= norm1[None,...]
                       
    if spikes2 is None:
        activity2, norm2 = activity1.copy(), norm1.copy()
    else:
        bins2 = cluster_spike_bins(spikes2,N_GR,N_GO,T)
        activity2 = compute_population_avg_activity(bins2, tau)
        norm2 = ((activity2**2).sum(axis=0))**.5
        activity2 /= norm2[None,...]

    # correlations
    c = dot(activity1.T,activity2)
    c1 = tile(c,[1,2])
    c2 = tile(c,[2,1])
    s1,s2 = [],[]
    
    # average over time per dt
    for i in xrange(c.shape[0]/2):
        s2.append(trace(c1,i)/(c.shape[0]))
        s1.append(trace(c2,-i)/(c.shape[0]))
        
    return hstack((flipud(array(s1)),array(s2)[1:]))
                