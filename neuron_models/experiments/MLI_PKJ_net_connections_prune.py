import datetime
import os
import gc
import multiprocessing
from itertools import repeat
from brian import *
import sys
sys.path.append('../../')
from neuron_models import *
import cPickle
import time
from itertools import product
set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)
defaultclock.dt = .25*ms

def fr_stats(spike_monitor):
    mean_frs = []
    for ind in range(len(spike_monitor.spiketimes)):
        isis = diff(spike_monitor.spiketimes[ind])
        if len(list(isis)) == 0:
            mean_frs.append((ind,0.))
        else:
            mean_frs.append((ind,mean(isis)**-1))
    return mean_frs

def isi_cv_stats(spike_monitor):
    cvs = []
    for ind in range(len(spike_monitor.spiketimes)):
        isi_mean, isi_std = isi_mean_and_std(spike_monitor, ind)
        cvs.append((ind,isi_std/isi_mean))
    return cvs

def prune_synapses(syn,pct):
    '''
    randomly set 'pct' percent synapse weights to 0 for synapse object 'syn'
    '''
    inds = random_integers(0,syn.w[:,:].shape[0]-1,int(syn.w[:,:].shape[0]*pct))
    syn.w[inds] = 0.

def run_net((syn_prune,prune_pct)):
    '''
    prune_pct: percent to prune the synapses by.
    '''
    seed(int(os.getpid()*time.time()))
    print os.getpid()
    reinit()
    reinit_default_clock()
    clear(True)
    gc.collect()

    T = 60*second
    N_MLI = 160
    N_PKJ = 16
    MLI = MLIGroup(N_MLI)
    PKJ = PurkinjeCellGroup(N_PKJ)

    # synaptic weights
    w_mli_pkj = 1.25
    w_mli_mli = 1.
    w_pkj_mli = 1.

    # Synapses
    S_MLI_PKJ = Synapses(MLI,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w')
    S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
    S_PKJ_MLI = Synapses(PKJ,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')

    # load saved synapses
    syn_dir = '/home/bill/shared_folder/research/paper #1/synapses/'
    S_MLI_PKJ = load_synapses(S_MLI_PKJ, 'S_MLI_PKJ', syn_dir)
    S_MLI_MLI = load_synapses(S_MLI_MLI, 'S_MLI_MLI', syn_dir)
    S_PKJ_MLI = load_synapses(S_PKJ_MLI, 'S_PKJ_MLI', syn_dir)

    if syn_prune == 'MLI_PKJ':
        prune_synapses(S_MLI_PKJ, prune_pct)
    elif syn_prune == 'MLI_MLI':
        prune_synapses(S_MLI_MLI, prune_pct)
    elif syn_prune == 'PKJ_MLI':
        prune_synapses(S_PKJ_MLI, prune_pct)
    else:
        raise Exception('syn_prune must be MLI_PKJ, MLI_MLI or PKJ_MLI')

    @network_operation(Clock(dt=defaultclock.dt))
    def random_current():
        MLI.I = gamma(3.966333,0.006653,size=len(MLI)) * nA
        PKJ.I = gamma(0.430303,0.195962,size=len(PKJ)) * nA

    # Monitor
    MS_MLI = SpikeMonitor(MLI)
    MS_PKJ = SpikeMonitor(PKJ)

    start = time.time()
    run(T)
    print time.time() - start

    return syn_prune, prune_pct, fr_stats(MS_MLI), isi_cv_stats(MS_MLI), fr_stats(MS_PKJ), isi_cv_stats(MS_PKJ)

if __name__ == "__main__":

    N_simulations = 6
    pool = multiprocessing.Pool(5)
    results = pool.map(run_net, product(['MLI_MLI', 'MLI_PKJ', 'PKJ_MLI'],[0.,.25,.5,.75,1.]))

    out_dir = '/home/bill/research/data/neuron_models/molecular_layer/clip_connections/'
    cPickle.dump(results,open(out_dir+'connections_prune_results.pkl','w'))


    
        