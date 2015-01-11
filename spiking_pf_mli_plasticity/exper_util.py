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

def connect_gr_mli(S_GR_MLI, N_MLI_groups=16,N_MLI_per_group=10, GR_cluster_width=3, N_GR_clusters=15, active_w=.2):
    '''
    :param S_GR_MLI: Synapse object connecting GR and MLI groups
    :param N_MLI_groups: the number of groups of MLIs (each group corresponds to a Purkinje cell)
    :param N_MLI_per_group: the number of MLIs in each group
    :param GR_cluster_width: the number of neurons wide a granule cell group that feeds input into an MLI is
    :param N_GR_clusters: the number of distinct clusters stacked on top of each other that make up the length
                          of the granule cell layer
    :param active_w: the initial value of the active synapses (can be a string passed to Brian Synapse.w)
    :return:synapse object
    '''
    GR_width = N_MLI_groups
    GR_length = GR_cluster_width*N_GR_clusters
    N_GR = GR_width * GR_length
    N_MLI = N_MLI_groups * N_MLI_per_group
    gr_active_start = N_GR_clusters*GR_cluster_width/2 - GR_cluster_width/2
    gr_active_end = N_GR_clusters*GR_cluster_width/2 + GR_cluster_width/2
    S_GR_MLI[:,:] = 'abs((i/GR_length)-(j/N_MLI_per_group))<=GR_cluster_width/2'
    pre_ind = [i for i in range(N_GR) if (gr_active_start<=(i%GR_length)<=gr_active_end)]
    S_GR_MLI.w[pre_ind,:] = active_w
    return S_GR_MLI

def setup_mli_net(pf_rates, cf_rates, init_pf_mli_w= '.05+.2*rand()', wmin=.2, N_CF=2,
                  N_MLI_groups=16,N_MLI_per_group=10, GR_cluster_width=3, N_GR_clusters=15):
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
    # compute parameters
    GR_width = N_MLI_groups
    GR_length = GR_cluster_width*N_GR_clusters
    N_GR = GR_width * GR_length
    N_MLI = N_MLI_groups * N_MLI_per_group

    # Neuron groups
    GR = PoissonGroup(N_GR,rates=pf_rates)
    CF = PoissonGroup(N_CF,rates=cf_rates)
    MLI = MLIGroup(N_MLI)

    # MLI-MLI Synapses
    S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
    S_MLI_MLI = connect_mli_mli(S_MLI_MLI,dist=80,syn_prob=.05)
    S_MLI_MLI.w[:,:] = 'rand()'

    # GR-MLI Synapses
    S_GR_MLI = Synapses(GR,MLI,model='''w:1
                        v:1''',pre='g_ampa_fast+=MLI.g_ampa_*v; g_ampa_slow+=MLI.g_ampa_*v; n+=1')
    S_GR_MLI = connect_gr_mli(S_GR_MLI,N_MLI_groups,N_MLI_per_group,GR_cluster_width,N_GR_clusters,init_pf_mli_w)
    ind = S_GR_MLI.w[:,:] > 0
    v_tmp =  S_GR_MLI.v[:,:]
    v_tmp[ind] = wmin + (1-wmin)*S_GR_MLI.w[:,:][ind]
    S_GR_MLI.v[:,:] = v_tmp

    # CF-MLI Synapses
    u,tau = .19, 2.1*second
    S_CF_MLI = Synapses(CF,MLI,model='''w:1
                        s:1''', pre='''s+=(1-s)*(1-exp(-(t-lastupdate)/tau)); g_ampa_fast+=MLI.g_ampa_*s*w;
                                       n+=3*s; s*=u''')
    S_CF_MLI[:,:] = True
    S_CF_MLI.w = .33
    S_CF_MLI.s = 1.

    return GR, CF, MLI, S_GR_MLI, S_CF_MLI

def setup_monitors_cf(GR, MLI, CF, S_GR_MLI):
    record_clock = Clock(1*ms)
    GR_S = SpikeMonitor(GR)
    GR_R = RealTimeRateMonitor(GR, record=True, record_clock=record_clock)
    MLI_V = StateMonitor(MLI, 'V', record=True, clock=record_clock)
    MLI_S = SpikeMonitor(MLI)
    MLI_R = RealTimeRateMonitor(MLI, tau_f=60*ms, tau_r=15*ms, record=True, record_clock=record_clock)
    CF_S = SpikeMonitor(CF)
    CF_R = RealTimeRateMonitor(CF, tau_f=20*ms, tau_r=2*ms, record=True, record_clock=record_clock)
    W = StateMonitor(S_GR_MLI,'v',record=True, clock=record_clock)
    return GR_S, GR_R, MLI_V, MLI_S, MLI_R, CF_S, CF_R, W
