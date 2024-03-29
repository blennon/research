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
    
    T = 4000
    N_MLI = 160
    N_PKJ = 16
    MLI = MLIGroup(N_MLI)
    PKJ = PurkinjeCellGroup(N_PKJ)
    
    # Synapses
    S_MLI_PKJ = Synapses(MLI,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w_mli_pkj')
    S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w_mli_mli')
    S_PKJ_MLI = Synapses(PKJ,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w_pkj_mli')
    S_PKJ_PKJ = Synapses(PKJ,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w_pkj_pkj')
    
    connect_mli_pkj(S_MLI_PKJ,pkj_dist=8,syn_prob=.125)
    connect_mli_mli(S_MLI_MLI,dist=80,syn_prob=.0275)
    S_PKJ_MLI[:,:] = 'j/(N_MLI/N_PKJ) == i'
    S_PKJ_PKJ[:,:] = '(minimum(abs(i-j),abs(abs(i-j)-N_PKJ))<=5) & (i!=j) & (rand()<.5)'
        
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
    
    mli_mew, mli_std = isi_mean_and_std(MS_MLI)
    pkj_mew, pkj_std = isi_mean_and_std(MS_PKJ)
    
    return mean(MR_MLI.rate), mli_std/mli_mew, mean(MR_PKJ.rate), pkj_std/pkj_mew

if __name__ == "__main__":
    pool = multiprocessing.Pool(7)
    w_pkj_pkj,w_pkj_mli,w_mli_mli,w_mli_pkj = linspace(0.1,2.,7),linspace(0.,1.,7),linspace(0.0,.2,7),linspace(0.0,1.,7)
    results = pool.map(run_net, itertools.product(w_pkj_pkj,w_pkj_mli,w_mli_mli,w_mli_pkj))
    
    out_dir = out_dir = '/home/bill/research/data/neuron_models/molecular_layer/mli_pkj_param_sweep/%s/'%datetime.datetime.now().isoformat()
    os.makedirs(out_dir)
    
    params = itertools.product(w_pkj_pkj,w_pkj_mli,w_mli_mli,w_mli_pkj)
    with open(out_dir+'results.txt','w') as outf:
        outf.write('\t'.join(['w_pkj_pkj','w_pkj_mli','w_mli_mli','w_mli_pkj','mli_mean_firing_rate','mli_cv','pkj_mean_firing_rate','pkj_cv'])+'\n')
        for r in results:
            param = params.next()
            outf.write('\t'.join(map(str,param))+'\t'+'\t'.join(map(str,r))+'\n')
        