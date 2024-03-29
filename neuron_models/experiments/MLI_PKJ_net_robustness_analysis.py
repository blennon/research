'''
The goal of this experiment is to simulate the network with slight perturbations to the parameters multiple times and
show that the network produces similar behavior.

This file dumps the data to disk. A corresponding IPython notebook is used to analyze the data,
"consistency_robustness_analysis.ipynb".
'''
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
set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)
defaultclock.dt = .25*ms

def perturb(x, pct=.1):
    '''perturb the value x by up to some percent pct, determined uniformly randomly'''
    return x + x*pct*(rand()-.5)/.5

def run_net(i):
    seed(i*int(os.getpid()*time.time()))
    print os.getpid()
    reinit()
    reinit_default_clock()
    clear(True)
    gc.collect()

    T = 30*second
    N_MLI = 160
    N_PKJ = 16
    MLI = MLIGroup(N_MLI)
    PKJ = PurkinjeCellGroup(N_PKJ)

    # synaptic weights
    w_mli_pkj = perturb(1.25)
    w_mli_mli = perturb(1.)
    w_pkj_mli = perturb(1.)

    # Synapses
    S_MLI_PKJ = Synapses(MLI,PKJ,model='w:1',pre='g_inh+=PKJ.g_inh_*w')
    S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')
    S_PKJ_MLI = Synapses(PKJ,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w')

    connect_mli_pkj(S_MLI_PKJ,pkj_dist=int(perturb(8)),syn_prob=perturb(.25))
    connect_mli_mli(S_MLI_MLI,dist=int(perturb(80)),syn_prob=perturb(.05))
    S_PKJ_MLI[:,:] = '((j/(N_MLI/N_PKJ)-i)%N_PKJ <= 2) & ((j/(N_MLI/N_PKJ)-i)%N_PKJ > 0) & (j%(N_MLI/N_PKJ)<3) & (rand()<perturb(.5))'
    S_MLI_PKJ.w[:,:] = 'rand()*w_mli_pkj'
    S_MLI_MLI.w[:,:] = 'rand()*w_mli_mli'
    S_PKJ_MLI.w[:,:] = 'rand()*w_pkj_mli'

    @network_operation(Clock(dt=defaultclock.dt))
    def random_current():
        MLI.I = gamma(perturb(3.966333),perturb(0.006653),size=len(MLI)) * nA
        PKJ.I = gamma(perturb(0.430303),perturb(0.195962),size=len(PKJ)) * nA

    # Monitor
    MS_MLI = SpikeMonitor(MLI)
    MS_PKJ = SpikeMonitor(PKJ)

    start = time.time()
    run(T)
    print time.time() - start

    return fr_stats(MS_MLI), isi_cv_stats(MS_MLI), fr_stats(MS_PKJ), isi_cv_stats(MS_PKJ)

if __name__ == "__main__":

    N_simulations = 100
    pool = multiprocessing.Pool(6)
    results = pool.map(run_net, range(N_simulations))
    cPickle.dump(results,open('robustness_results.pkl','w'))


    
        