__author__ = 'bill'
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