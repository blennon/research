import gc
import os
import multiprocessing
import cPickle
import time
from brian import *
from neuron_models import *
set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)#, gcc_options=['-ffast-math', '-march=native'])
defaultclock.dt = .5*ms


def run_net((T,proc_num)):
    seed(os.getpid()*proc_num)
    reinit()
    reinit_default_clock()
    clear(True)
    gc.collect()
    
    # Neurons
    GO = GolgiCellGroup(N_GO)
    GR = GranuleCellGroup(N_GR)
    PG = PoissonGroup(N_PG, lambda t: 15 * Hz + 30 * Hz * 0.5 * (1 + cos(2 * pi * t * 1.5 * Hz + pi)))
    
    # Synapses
    w_gr_go = .2/(49*(len(GR)/len(GO)))
    w_go_gr = 10. * 2
    w_mf_gr = 4.
    S_GO_GR = Synapses(GO,GR,model='w:1',pre='g_inh1+=GR.g_inh_*w_go_gr; g_inh2+=GR.g_inh_*w_go_gr')
    S_GR_GO = Synapses(GR,GO,model='w:1',pre='g_ampa+=GO.g_ampa_*w_gr_go;g_nmda1+=GO.g_ampa_*w_gr_go;g_nmda2+=GO.g_ampa_*w_gr_go')
    S_PG_GR = Synapses(PG,GR,model='w:1',pre='g_ampa+=GR.g_ampa_*w_mf_gr;g_nmda+=GR.g_ampa_*w_mf_gr')
    S_PG_GR.connect_one_to_one()
    for src,trg in gr_to_go:
        S_GR_GO[src,trg] = 1.
    for src,trg in go_to_gr:
        S_GO_GR[src,trg] = 1.
    
    # Monitor
    MS_PG = SpikeMonitor(PG)
    MS_GR = SpikeMonitor(GR)
    MS_GO = SpikeMonitor(GO)
    
    start = time.time()
    run(T)
    print time.time() - start
    
    return MS_GR.spiketimes

if __name__ == "__main__":
    out_dir = '/home/bill/research/data/neuron_models/granule_layer/'
    N_GO = 32**2
    N_GR = N_GO * 5**2
    N_PG = N_GR
    gr_to_go = gr_to_go_connections(N_GO,N_GR)
    go_to_gr = go_to_gr_connections(N_GO,N_GR)
    
    pool = multiprocessing.Pool(1)
    
    results = pool.map(run_net, [(2*second,i) for i in range(100)])
    cPickle.dump(results, open(out_dir+'granule_layer_par_100runs_3s_041113','w'))
