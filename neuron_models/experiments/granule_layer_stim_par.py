'''
This experiment creates a network and excites it with various
stimuli wave forms (sinusoidal, triangle, square)

The output is then saved for later analysis
'''
import gc
import os
import multiprocessing
import cPickle
import time
from brian import *
from neuron_models import *
set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)
defaultclock.dt = .5*ms


def run_net((stimulus_name,T,proc_num)):
    seed(os.getpid()*proc_num)
    reinit()
    reinit_default_clock()
    clear(True)
    gc.collect()
    
    stimuli = {'triangle_two_periods':lambda t: stimulus(t,15*Hz,30*Hz,(1/1.)* Hz,0.,triangle_wave),
               'square_two_periods_in_phase':lambda t: stimulus(t,15*Hz,30*Hz,(1/1.)* Hz,.25,square_wave),
               'sinusoid_two_periods':lambda t: stimulus(t,15*Hz,15*Hz,2*pi*Hz,pi,lambda x: 1+cos(x)),
               'triangle_one_period':lambda t: stimulus(t,15*Hz,30*Hz,(1/2.)* Hz,0.,triangle_wave),
               'square_one_period_in_phase':lambda t: stimulus(t,15*Hz,30*Hz,(1/2.)* Hz,0.5,square_wave),
               'sinusoid_one_period':lambda t: stimulus(t,15*Hz,15*Hz,pi*Hz,pi,lambda x: 1+cos(x))
               }
    
    # Neurons
    GO = GolgiCellGroup(N_GO)
    GR = GranuleCellGroup(N_GR)
    stimulus_f = stimuli[stimulus_name]
    PG = PoissonGroup(N_PG, stimulus_f)
    
    # Synapses
    S_GO_GR = Synapses(GO,GR,model='w:1',pre='g_inh1+=GR.g_inh_*w_go_gr; g_inh2+=GR.g_inh_*w_go_gr')
    S_GR_GO = Synapses(GR,GO,model='w:1',pre='g_ampa+=GO.g_ampa_*w_gr_go;g_nmda1+=GO.g_ampa_*w_gr_go;g_nmda2+=GO.g_ampa_*w_gr_go')
    S_PG_GR = Synapses(PG,GR,model='w:1',pre='g_ampa+=GR.g_ampa_*w_mf_gr;g_nmda+=GR.g_ampa_*w_mf_gr')
    S_PG_GR.connect_one_to_one()
    for src,trg in gr_to_go:
        S_GR_GO[src,trg] = 1.
    for src,trg in go_to_gr:
        S_GO_GR[src,trg] = 1.
    
    # Monitor
    #MS_PG = SpikeMonitor(PG)
    MS_GR = SpikeMonitor(GR)
    #MS_GO = SpikeMonitor(GO)
    
    start = time.time()
    run(T)
    print time.time() - start
    
    return MS_GR.spiketimes, stimulus_name

if __name__ == "__main__":
    out_dir = '/home/bill/research/data/neuron_models/granule_layer/stimuli_par/%s/'%datetime.datetime.now().isoformat()
    T = 2000
    N_GO = 32**2
    N_GR = N_GO * 10**2
    N_PG = N_GR
    w_gr_go = .2/(49*(N_GR/N_GO))
    w_go_gr = 10. * 2
    w_mf_gr = 4.
    gr_to_go = gr_to_go_connections(N_GO,N_GR)
    go_to_gr = go_to_gr_connections(N_GO,N_GR)
    
    pool = multiprocessing.Pool(2)

    #stimuli_names = ['triangle_two_periods','square_two_periods_in_phase','sinusoid_two_periods',
    #                 'triangle_one_period','square_one_period_in_phase','sinusoid_one_period']
    stimuli_names = ['sinusoid_one_period','sinusoid_one_period']
    params = []
    i = 1
    for n in stimuli_names:
        params.append((n,T*msecond,i))
        i += 1

    results = pool.map(run_net, params)
    
    os.makedirs(out_dir)
    j=0
    for GR_spikes, stim in results:
        j+=1
        with open(out_dir+stim+str(j)+'_spikes.pickle','w') as outf:
            cPickle.dump(GR_spikes, outf)
            
        close('all')
        # plot and save
        figs = []
        # similarity
        fig2 = figure(2)
        sim = population_spike_correlation(GR_spikes,GR_spikes,N_GR,N_GO,T)
        plot(arange(-T/2,T/2-1),sim)
        xlabel('time delta')
        ylabel('Similarity')
        title('Granule Cell Similarity Measure')
        figs.append(fig2)
        i = 1
        for fig in figs:
            try:
                fig.savefig(out_dir+stim+'fig%s'%i)
            except RuntimeError:
                pass
            i += 1         
            
    with open(out_dir+'weights','w') as outf:
        outf.write('w_gr_go %s\n'%w_gr_go)
        outf.write('w_go_gr %s\n'%w_go_gr)
        outf.write('w_mf_gr %s\n'%w_mf_gr)
    GO = GolgiCellGroup(N_GO)
    GR = GranuleCellGroup(N_GR)
    with open(out_dir+'GR','w') as outf:
        GR.save_parameters(outf)
        outf.close()
    with open(out_dir+'GO','w') as outf:
        GO.save_parameters(outf)
        outf.close()
    #cPickle.dump(results, open(out_dir+'granule_layer_par_100runs_3s_041113','w'))
