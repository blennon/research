__author__ = 'bill'
from brian import *
from neuron_models import *
from spiking_pf_mli_plasticity.rate_monitor import *

def setup_monitors(GR, MLI, S_GR_MLI):
    record_clock = Clock(1*ms)
    GR_S = SpikeMonitor(GR)
    GR_R = RealTimeRateMonitor(GR, record=True, record_clock=record_clock)
    MLI_V = StateMonitor(MLI, 'V', record=True, clock=record_clock)
    MLI_S = SpikeMonitor(MLI)
    MLI_R = RealTimeRateMonitor(MLI, tau_f=60*ms, tau_r=15*ms, record=True, record_clock=record_clock)
    W = StateMonitor(S_GR_MLI,'v',record=True, clock=record_clock)
    return GR_S, GR_R, MLI_V, MLI_S, MLI_R, W

def setup_isolated_mli_net(N_GR, initial_weights, pf_rates, wmin):
    '''
    sets up the network consisting of a single MLI and GR inputs
    :param N_GR: number of granule cells
    :param initial_weights: array, initial weights for PF-MLI synapses
    :param pf_rates: a function that returns the firing rates of GRs as a function of time
    :param wmin: the minimum weight value allowed for PF-MLI synapses, [0,1)
    :return: GR neuron group, MLI neuron group, GR-MLI synapses object
    '''
    GR = PoissonGroup(N_GR,rates=pf_rates)
    MLI = MLIGroup(1)
    #S_GR_MLI = Synapses(GR,MLI,model='''w:1
    #                    v:1''',pre='g_ampa_fast+=MLI.g_ampa_*v; g_ampa_slow+=MLI.g_ampa_*v; g_nmda+=MLI.g_nmda_*v')
    S_GR_MLI = Synapses(GR,MLI,model='''w:1
                        v:1''',pre='g_ampa_fast+=MLI.g_ampa_*v; g_ampa_slow+=MLI.g_ampa_*v; n+=1')
    S_GR_MLI[:,:] = 1
    S_GR_MLI.w[:,:] = initial_weights
    S_GR_MLI.v[:,:] = wmin + (1-wmin)*S_GR_MLI.w[:,:]
    return GR, MLI, S_GR_MLI

def setup_independent_isolated_mli_nets(N_GR, initial_weights, pf_rates, wmin):
    '''
    sets up several PF-MLI networks in parallel for simulation. I.e. N_GR=N_MLI
    and one PF uniquely contacts one MLI (no crossover). Thus, these are independent
    experiments
    :param N_GR: number of granule cells -- and number of MLIs
    :param initial_weights: array, initial weights for PF-MLI synapses
    :param pf_rates: a function that returns the firing rates of GRs as a function of time
    :param wmin: the minimum weight value allowed for PF-MLI synapses, [0,1)
    :return: GR neuron group, MLI neuron group, GR-MLI synapses object
    '''
    GR = PoissonGroup(N_GR,rates=pf_rates)
    MLI = MLIGroup(N_GR)
    S_GR_MLI = Synapses(GR,MLI,model='''w:1
                        v:1''',pre='g_ampa_fast+=MLI.g_ampa_*v; g_ampa_slow+=MLI.g_ampa_*v; n+=1')
    S_GR_MLI.connect_one_to_one()
    S_GR_MLI.w[:,:] = initial_weights
    S_GR_MLI.v[:,:] = wmin + (1-wmin)*S_GR_MLI.w[:,:]
    return GR, MLI, S_GR_MLI

def build_pf_rates_func(eq_time, trial_time, N_GR, base_rate, stim_rate, stim_len):
    '''
    :param eq_time: the time at the beginning of the experiment to allow GRs to fire
                    at baseline activity
    :param trial_time: the length of the trial
    :param N_GR: number of granule cells
    :param base_rate: the baseline firing rate of GRs
    :param stim_rate: the desired firing rate during stimulation of PFs
    :return:function of time that returns pf rates
    '''
    def pf_rates(t):
        '''
        parallel fire input stimulus -- Poisson rates

        t: the current simulation time step
        '''
        if t>=eq_time and t%(trial_time)<stim_len:
            return ones(N_GR)*stim_rate
        return ones(N_GR)*base_rate
    return pf_rates