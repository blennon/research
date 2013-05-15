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
defaultclock.dt = .5*ms

def isi_mean_and_std(monitor):
    '''
    compute the mean and variance of interspike intervals
    of a group of neurons
    '''
    isi = []
    for n_ind, times in monitor.spiketimes.iteritems():
        isi += list(diff(times)*1000)
    return mean(isi), var(isi)**.5

def run_net((w_pkj_,w_bs_)):
    seed(os.getpid())
    print os.getpid()
    reinit()
    reinit_default_clock()
    clear(True)
    gc.collect()
    
    T = 60000
    N_BS = 16
    N_PKJ = 16
    BS = BasketCellGroup(N_BS)
    PKJ = PurkinjeCellGroup(N_PKJ)
    
    # synaptic weights
    w_bs_pkj = w_bs_
    w_bs_bs = w_bs_
    w_pkj_bs = w_pkj_
    w_pkj_pkj = w_pkj_
    
    # Synapses
    S_BS_PKJ = Synapses(BS,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w_bs_pkj')
    S_BS_BS = Synapses(BS,BS,model='w:1',pre='g_inh+=BS.g_inh_*w_bs_bs')
    S_PKJ_BS = Synapses(PKJ,BS,model='w:1',pre='g_inh+=BS.g_inh_*w_pkj_bs')
    S_PKJ_PKJ = Synapses(PKJ,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w_pkj_pkj')
    
    # Connections
    S_BS_PKJ[:,:] = 'minimum(abs(i-j),abs(abs(i-j)-N_PKJ))<=1' 
    S_BS_BS[:,:] = '(minimum(abs(i-j),abs(abs(i-j)-N_BS))<=1) & (i!=j)'
    S_PKJ_BS.connect_one_to_one()
    S_PKJ_PKJ[:,:] = '(minimum(abs(i-j),abs(abs(i-j)-N_PKJ))<=1) & (i!=j)'
    
    @network_operation(Clock(dt=1*ms))
    def random_current():
        bs_i = .0635 + .045*randn(len(BS))
        bs_i[bs_i<0] = 0.0
        BS.I = bs_i * nA
        pkj_i = .08 + .075*randn(len(PKJ))
        pkj_i[pkj_i<0] = 0.
        PKJ.I = pkj_i * nA
        
    # Monitor
    MS_BS = SpikeMonitor(BS)
    MR_BS = PopulationRateMonitor(BS,bin=1*ms)
    MISI_BS = ISIHistogramMonitor(BS,bins=arange(0,120,2)*ms)
    
    MS_PKJ = SpikeMonitor(PKJ)
    MR_PKJ = PopulationRateMonitor(PKJ,bin=1*ms)
    MISI_PKJ = ISIHistogramMonitor(PKJ,bins=arange(0,120,2)*ms)
    
    start = time.time()
    run(T*msecond)
    print time.time() - start
    
    bs_mew, bs_std = isi_mean_and_std(MS_BS)
    pkj_mew, pkj_std = isi_mean_and_std(MS_PKJ)
    
    return mean(MR_BS.rate), bs_std/bs_mew, mean(MR_PKJ.rate), pkj_std/pkj_mew

if __name__ == "__main__":
    pool = multiprocessing.Pool(6)
    results = pool.map(run_net, itertools.product(linspace(.5,3,6),linspace(.5,3,6)))
    
    out_dir = out_dir = '/home/bill/research/data/neuron_models/molecular_layer/bs_pkj_param_sweep/%s/'%datetime.datetime.now().isoformat()
    os.makedirs(out_dir)
    
    params = itertools.product(linspace(.5,3,6),linspace(.5,3,6))
    with open(out_dir+'results.txt','w') as outf:
        outf.write('\t'.join(['w_pkj_','w_bs_','bs_mean_firing_rate','bs_cv','pkj_mean_firing_rate','pkj_cv'])+'\n')
        for r in results:
            param = params.next()
            outf.write('\t'.join(map(str,param))+'\t'+'\t'.join(map(str,r))+'\n')
        