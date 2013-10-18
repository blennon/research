'''
This script performs a grid search for the current parameters that best match the data
from Hausser and Clark (1997)
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

def isi_mean_and_std(monitor):
    '''
    compute the mean and variance of interspike intervals
    of a group of neurons
    '''
    isi = []
    for n_ind, times in monitor.spiketimes.iteritems():
        isi += list(diff(times)*1000)
    return mean(isi), var(isi)**.5

def run_net((k,theta)):
    seed(os.getpid())
    print os.getpid()
    reinit()
    reinit_default_clock()
    clear(True)
    gc.collect()
    
    T = 6000
    N_PKJ = 1
    PKJ = PurkinjeCellGroup(N_PKJ)
        
    @network_operation(Clock(dt=defaultclock.dt))
    def random_current():
        PKJ.I = gamma(k,theta,size=len(PKJ)) * nA
            
    # Monitor
    MS_PKJ = SpikeMonitor(PKJ)
    MR_PKJ = PopulationRateMonitor(PKJ,bin=1*ms)
    MISI_PKJ = ISIHistogramMonitor(PKJ,bins=arange(0,162,2)*ms)
    
    start = time.time()
    run(T*msecond)
    print time.time() - start
    
    PKJ_mew, PKJ_std = isi_mean_and_std(MS_PKJ)
    
    return k,theta,mean(MR_PKJ.rate), PKJ_std/PKJ_mew

if __name__ == "__main__":
    pool = multiprocessing.Pool(6)
    params = []
    for k in linspace(.1,11,100):
        for theta in linspace(.0001,.2,100):
            if k*theta < .2 and k*theta > .03:
                params.append((k,theta))
    print len(params)
    results = pool.map(run_net, params)
    
    out_dir = out_dir = '/home/bill/research/data/neuron_models/molecular_layer/PKJ_gamma_current_param_sweep/%s/'%datetime.datetime.now().isoformat()
    os.makedirs(out_dir)
    
    with open(out_dir+'results.txt','w') as outf:
        outf.write('\t'.join(['k','theta','PKJ_mean_firing_rate','PKJ_cv'])+'\n')
        for r in results:
            outf.write('\t'.join(map(str,r))+'\n')
        