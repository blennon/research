'''
This script performs a parameter sweep over the weight/conductance values of connections
among a network of MLIs and PKJs.  It measures the error of the fit of mean and CV of the
ISIs by finding the single neuron among MLIs and PKJs that is closest to the prototypes
reported by Hausser and Clark (1997).
'''
import datetime
import os
import gc
import multiprocessing
import itertools
from brian import *
import sys
from neuron_models import *
import cPickle
import time
set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)
defaultclock.dt = .25*ms


def run_net((w_pkj_pkj,w_pkj_mli,w_mli_mli,w_mli_pkj)):
    seed(int(os.getpid()*time.time()))
    print os.getpid()
    reinit()
    reinit_default_clock()
    clear(True)
    gc.collect()
    
    T = 6000
    N_MLI = 160
    N_PKJ = 16
    MLI = MLIGroup(N_MLI)
    PKJ = PurkinjeCellGroup(N_PKJ)
    
    # Synapses
    S_MLI_PKJ = Synapses(MLI,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w')
    S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
    S_PKJ_MLI = Synapses(PKJ,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
    S_PKJ_PKJ = Synapses(PKJ,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w')
    
    # Connections
    connect_mli_pkj(S_MLI_PKJ,pkj_dist=8,syn_prob=.25)
    connect_mli_mli(S_MLI_MLI,dist=80,syn_prob=.05)
    S_PKJ_MLI[:,:] = '(j/(N_MLI/N_PKJ) == i) & (rand()<.5)'
    S_PKJ_PKJ[:,:] = '(minimum(abs(i-j),abs(abs(i-j)-N_PKJ))<=5) & (i!=j) & (rand()<.25)'
    S_MLI_PKJ.w[:,:] = 'rand()*w_mli_pkj'
    S_MLI_MLI.w[:,:] = 'rand()*w_mli_mli'
    S_PKJ_MLI.w[:,:] = 'rand()*w_pkj_mli'
    S_PKJ_PKJ.w[:,:] = 'rand()*w_pkj_pkj'
        
    @network_operation(Clock(dt=defaultclock.dt))
    def random_current():
        MLI.I = gamma(3.966333,0.006653,size=len(MLI)) * nA
        PKJ.I = gamma(0.430303,0.195962,size=len(PKJ)) * nA
            
    # Monitor
    MS_MLI = SpikeMonitor(MLI)
    MR_MLI = PopulationRateMonitor(MLI,bin=1*ms)
    MISI_MLI = ISIHistogramMonitor(MLI,bins=arange(0,162,2)*ms)
    
    MS_PKJ = SpikeMonitor(PKJ)
    MR_PKJ = PopulationRateMonitor(PKJ,bin=1*ms)
    MISI_PKJ = ISIHistogramMonitor(PKJ,bins=arange(0,162,2)*ms)
    
    start = time.time()
    run(T*msecond)
    print time.time() - start
    
    mli_ind, mli_mean_fr, mli_isi_cv, mli_error = find_closest_match_neuron(MS_MLI, 15., .4)
    pkj_ind, pkj_mean_fr, pkj_isi_cv, pkj_error = find_closest_match_neuron(MS_PKJ, 35., .49)
    
    return mli_mean_fr, mli_isi_cv, pkj_mean_fr, pkj_isi_cv

if __name__ == "__main__":
    pool = multiprocessing.Pool(6)
    w_pkj_pkj,w_pkj_mli,w_mli_mli,w_mli_pkj = linspace(0.1,1.,6),linspace(0.1,1.,6),linspace(0.1,1.,6),linspace(0.1,1.,6)
    results = pool.map(run_net, itertools.product(w_pkj_pkj,w_pkj_mli,w_mli_mli,w_mli_pkj))
    
    out_dir = out_dir = '/home/bill/research/data/neuron_models/molecular_layer/mli_pkj_param_sweep3/%s/'%datetime.datetime.now().isoformat()
    os.makedirs(out_dir)
    
    params = itertools.product(w_pkj_pkj,w_pkj_mli,w_mli_mli,w_mli_pkj)
    with open(out_dir+'results.txt','w') as outf:
        outf.write('\t'.join(['w_pkj_pkj','w_pkj_mli','w_mli_mli','w_mli_pkj','mli_mean_firing_rate','mli_cv','pkj_mean_firing_rate','pkj_cv'])+'\n')
        for r in results:
            param = params.next()
            outf.write('\t'.join(map(str,param))+'\t'+'\t'.join(map(str,r))+'\n')
        