__author__ = 'bill'
from pylab import *
from brian import *

def synapse_inds(neuron_inds, synapses, neuron_inds_position='pre'):
    '''
    takes a 'synapses' object and a set of pre/post synaptic neuron
    indices, returns a sorted array of post/pre synapse indices and
    corresponding sorted array of pre/post synaptic neuron indices
    '''
    if neuron_inds_position == 'post':
        synapses_darr = synapses.synapses_post
    elif neuron_inds_position == 'pre':
        synapses_darr = synapses.synapses_pre
    else:
        raise Exception("Which must be 'pre' or 'post'")
    syns = array(synapses_darr)[neuron_inds]
    syns_inds = hstack(syns)
    neuron_inds = hstack(i*ones(s.shape[0],dtype=int) for i,s in enumerate(syns))
    sort_inds = argsort(syns_inds)
    return syns_inds[sort_inds], neuron_inds[sort_inds]

def compute_alpha(CF_R, S_CF_MLI, CF_max=10*Hz, alpha0=.5, alpha1=1.):
    '''
    computes 'alpha' in the learning equation for PF-MLI weight updates for each MLI.
    First computes an array which measures the total glutamate spillover from CFs onto each MLI.
    This is then converted to alpha which is a monotonically decreasing function of the measure
    of glutamate spillover / CF activity.

    CF_R: RateMonitor for CFs
    S_CF_MLI: synapses object for CF-MLI synapses
    CF_max: maximum CF firing rate
    alpha0: minimum value to return for alpha.

    returns an array with size the number of MLI neurons. Each value in the array is a measure between [alpha0,1]
    '''
    cf_fr = CF_R.get_normalized_firing_rates(CF_max)
    C = ceil(nan_to_num(S_CF_MLI.w.to_matrix()))
    spillover = minimum(dot(C.T, cf_fr),1.)
    return alpha1 - (alpha1-alpha0)*spillover

def update_weights(S_GR_MLI, GR_R, MLI_R, wmin, GR_max=500*Hz, MLI_max=150*Hz, beta=.001, alpha=1.):
    '''
    S_GR_MLI: synapses object between GR and MLI NeuronGroups
    GR_R: RateMonitor for GRs
    MLI_R: RateMonitor for MLIs
    wmin: minimum weight value
    GR_max: maximum firing rate for GRs to compute normalized firing rate
    MLI_max: maximum firing rate for MLIs to compute normalized firing rate
    beta: constant learning rate parameter
    alpha: learning parameter, possibly dynamic
    '''
    syn_inds, pre_inds = synapse_inds(range(len(GR_R)),S_GR_MLI,'pre')
    _, post_inds = synapse_inds(range(len(MLI_R)),S_GR_MLI,'post')
    mli_fr = MLI_R.get_normalized_firing_rates(MLI_max)
    gr_fr = GR_R.get_normalized_firing_rates(GR_max)
    S_GR_MLI.w[:,:] += beta*gr_fr[pre_inds]*(mli_fr[post_inds] - alpha*S_GR_MLI.w[:,:])
    S_GR_MLI.v[:,:] = wmin + (1-wmin)*S_GR_MLI.w[:,:]

def update_weights_cf(S_GR_MLI, S_CF_MLI, GR_R, MLI_R, CF_R, wmin, alpha0=.5, alpha1=1., alpha_thresh=.8,
                      gr_thresh=.1, CF_max=10*Hz, GR_max=500*Hz, MLI_max=150*Hz, beta=.001):
    '''
    Implements the full learning rule for PF-MLI synapses which is dependent on CF input.

    S_GR_MLI: synapses object between GR and MLI NeuronGroups
    S_CF_MLI: synapses object between CF and MLI NeuronGroups
    GR_R: RateMonitor for GRs
    MLI_R: RateMonitor for MLIs
    CF_R: RateMonitor for CFs
    wmin: minimum weight value for electrically active synapses
    alpha0: minimum value for alpha
    alpha_thresh: threshold value of alpha used for activating silent PF-MLI synapses
    gr_thresh: threshold value of GR firing rates used for activating silent PF-MLI synapses
    CF_max: maximum CF firing rate to compute normalized firing rate
    GR_max: maximum firing rate for GRs to compute normalized firing rate
    MLI_max: maximum firing rate for MLIs to compute normalized firing rate
    beta: constant learning rate parameter
    '''
    syn_inds, pre_inds = synapse_inds(range(len(GR_R)),S_GR_MLI,'pre')
    _, post_inds = synapse_inds(range(len(MLI_R)),S_GR_MLI,'post')
    mli_fr = MLI_R.get_normalized_firing_rates(MLI_max)
    gr_fr = GR_R.get_normalized_firing_rates(GR_max)
    alpha = compute_alpha(CF_R, S_CF_MLI, CF_max, alpha0, alpha1)

    # CF-PF gated LTP
    # for weights == 0 and CF inputs > 0 (use alpha as surrogate) and PF inputs > 0
    ind = (S_GR_MLI.v[:] == 0.) & (alpha[post_inds] < alpha_thresh) & (gr_fr[pre_inds] > gr_thresh)
    w_tmp = S_GR_MLI.w[:]
    w_tmp[ind] = .1*wmin

    # GSD
    # for weights > 0
    ind = w_tmp > 0
    w_tmp[ind] += beta*gr_fr[pre_inds][ind]*(mli_fr[post_inds][ind] - alpha[post_inds][ind]*w_tmp[ind])

    # set minimum weights and maximum weights (due to alpha)
    w_tmp[w_tmp > 1.] = 1. # hard max bound on weights
    v_tmp = S_GR_MLI.v[:]
    v_tmp[ind] = wmin + (1-wmin)*w_tmp[ind]
    ind = w_tmp < .05*wmin
    w_tmp[ind] = 0.
    v_tmp[ind] = 0.

    # Assign temporary variable to weights
    S_GR_MLI.w[:] = w_tmp
    S_GR_MLI.v[:] = v_tmp
