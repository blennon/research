'''
the goal of this experiment is to reproduce the results of Figure 5 
from Hausser and Clark (1997) where the effect of an IPSC from a 
presynaptic MLI increases the variance of the PKJ ISI
'''
import datetime
import os
import gc
import multiprocessing
from itertools import repeat
from brian import *
import sys
from neuron_models import *
import cPickle
import time
set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)
defaultclock.dt = .25*ms

def run_net((k,theta,T,g_inh_,spike_delay)):
    seed(int(os.getpid()*time.time()))
    print os.getpid()
    reinit()
    reinit_default_clock()
    clear(True)
    gc.collect()
    
    g_inh = g_inh_*rand()
    
    PKJ = PurkinjeCellGroup(1)
    PKJ.V = PKJ.El
    
    spikes = SpikeMonitor(PKJ)
    spikes.last_spike = None
    V_monitor = StateMonitor(PKJ,'V',record=0)
    ginh_monitor = StateMonitor(PKJ, 'g_inh', record=0)
    
    @network_operation(Clock(dt=defaultclock.dt))
    def random_current():
        PKJ.I = gamma(k,theta,size=len(PKJ)) * nA
        
    @network_operation(Clock(dt=defaultclock.dt))
    def trigger_spike():
        if spikes.spiketimes[0].shape[0] > 0:
            spikes.last_spike = spikes.spiketimes[0][-1]*second
        if spikes.last_spike is not None:
            if abs(defaultclock.t - (spikes.last_spike + spike_delay)) < .000001*ms:
                PKJ.g_inh = g_inh
        
    run(T)

    V_monitor.insert_spikes(spikes)
    first_isi = diff(spikes.spiketimes[0])[0]
    
    return V_monitor.getvalues(), first_isi, spikes.spiketimes, g_inh

if __name__ == "__main__":
    k, theta, T, g_inh_max, delay = 0.430303, 0.195962, .08*second, 10*nS, 12*ms
    params = tuple([k,theta,T,g_inh_max,delay])
    plist = []
    for i in range(1000):
        plist.append(params)
    pool = multiprocessing.Pool(6)
    results = pool.map(run_net, plist)
    
    out_dir = '/home/bill/research/data/neuron_models/molecular_layer/MLI_PKJ_ISI_delay_rand_nS/%s/' % datetime.datetime.now().isoformat()
    os.makedirs(out_dir)
    
    # write parameters to file
    with open(out_dir+'parameters.txt','w') as outf:
        outf.write('\t'.join(['k','theta','T','g_inh_max','delay','dt'])+'\n')
        outf.write('\t'.join(map(str,params))+'\t'+str(defaultclock.dt))
    
    # write voltage traces, isis and spike times
    cPickle.dump([r[0][0] for r in results],open(out_dir+'traces.pkl','w'))
    cPickle.dump([r[1] for r in results],open(out_dir+'isis.pkl','w'))
    cPickle.dump([r[2][0] for r in results],open(out_dir+'spikes.pkl','w'))
    cPickle.dump([r[3] for r in results],open(out_dir+'g_inh.pkl','w'))
    
        