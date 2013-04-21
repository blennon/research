from pylab import *
from brian import *
from abstract_neuron_group import AbstractNeuronGroup

class GranuleCellGroup(AbstractNeuronGroup):
    '''
    Group of granule cells
    '''
    Vth = -35 * mvolt            # Firing threshold, Volts
    Cm = 3.1 * pfarad            # Membrane capacitance

    El = -58 * mvolt             # leak reversal potential
    Eex = 0. * mvolt             # Excitatory reversal potential
    Einh = -82. * mvolt          # Inhibitory reversal potential
    Eahp = -82. * mvolt          # After hyperpolarization reversal potential

    gl = 0.43 * nsiemens         # maximum leak conductance
    g_ampa_ = 0.18 * nsiemens    # maximum ampa conductance
    g_nmda_ = 0.025 * nsiemens   # maximum nmda conductance
    g_inh_ = 0.028 * nsiemens    # maximum inhibitory conductance
    gahp_ = 1. * nsiemens        # maximum after hyperpolarization conductance
    
    tau_ampa = 1.2 * msecond     # AMPA time constant
    tau_nmda = 52. * msecond     # NMDA time constant
    tau_inh1 = 7. * msecond      # Inhbitory time constant 1
    tau_inh2 = 59. * msecond     # Inhbitory time constant 2
    tau_ahp = 5. * msecond       # AHP time constant
    
    eqns = Equations('''
    # Membrane equation
    dV/dt = 1/Cm*(-gl*(V-El)-(.88*g_ampa+.12*g_nmda)*(V-Eex)-g_inh*(V-Einh)-gahp*(V-Eahp)) : mV
    
    # After hyperpolarization
    dgahp/dt = -gahp/tau_ahp : nS
    
    # Glutamate
    dg_ampa/dt = -g_ampa/tau_ampa : nS
    dg_nmda/dt = -g_nmda/tau_nmda : nS
    
    # GABA
    dg_inh1/dt = -g_inh1/tau_inh1 : nS
    dg_inh2/dt = -g_inh2/tau_inh2 : nS
    g_inh = .43 * g_inh1 + .57 * g_inh2 : nS
    ''')
    
    def __init__(self, N, rand_V_init = True):
        
        super(GranuleCellGroup, self).__init__(N, model=GranuleCellGroup.eqns,threshold=GranuleCellGroup.Vth,
                                               reset='V=GranuleCellGroup.El;gahp=GranuleCellGroup.gahp_')

        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Einh':self.Einh,'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_,
                  'g_nmda_':self.g_nmda_,'g_inh_':self.g_inh_,'gahp_':self.gahp_,
                  'tau_ampa':self.tau_ampa,'tau_nmda':self.tau_nmda,'tau_inh1':self.tau_inh1,
                  'tau_inh2':self.tau_inh2,'tau_ahp':self.tau_ahp,'eqns':self.eqns}
        return params


class GolgiCellGroup(AbstractNeuronGroup):
    '''
    Group of Golgi cells
    '''
    Vth = -52. * mvolt          # Firing threshold, Volts
    Cm = 28. * pfarad           # Membrane capacitance

    El = -55. * mvolt           # leak reversal potential
    Eex = 0. * mvolt            # Excitatory reversal potential
    Einh = -82. * mvolt         # Inhibitory reversal potential
    Eahp = -72.7 * mvolt        # After hyperpolarization reversal potential

    gl = 2.3 * nsiemens         # maximum leak conductance
    g_ampa_ = 45.5 * nsiemens   # maximum ampa conductance
    g_nmda_ = 30. * nsiemens    # maximum nmda conductance
    gahp_ = 20. * nsiemens      # maximum after hyperpolarization conductance
    

    tau_ampa = 1.5 * msecond    # AMPA time constant, 2.05 is approximately equivalent to Yamazaki's 1.5
                                # Integration methods are sensitive to short time constants.  Yamazaki multiplies
                                # values that decay by -x/tau by exp(-dt/tau) instead of performing exact integration
                                # of the differential equations (as BRIAN does).  Yamazaki's method is sensitive to 
                                # shorter time constants, having greater error the shorter the time constants.
    tau_nmda1 = 31. * msecond   # NMDA time constant 1
    tau_nmda2 = 170. * msecond  # NMDA time constant 2    
    tau_ahp = 5. * msecond      # AHP time constant
    
    eqns = Equations('''
    # Membrane equation
    dV/dt = 1/Cm*(-gl*(V-El)-g_ex*(V-Eex)-gahp*(V-Eahp)) : mV
    
    # After hyperpolarization
    dgahp/dt = -gahp/tau_ahp : nS
    
    # Glutamate
    dg_ampa/dt = -g_ampa/tau_ampa : nS
    dg_nmda1/dt = -g_nmda1/tau_nmda1 : nS
    dg_nmda2/dt = -g_nmda2/tau_nmda2 : nS
    g_nmda = .33 * g_nmda1 + .67 * g_nmda2 : nS
    g_ex = .8 * g_ampa + .2 * g_nmda : nS
    ''')
    
    def __init__(self, N, rand_V_init = True):
        super(GolgiCellGroup, self).__init__(N, model=GolgiCellGroup.eqns,threshold=GolgiCellGroup.Vth,
                                               reset='V=GolgiCellGroup.El;gahp=GolgiCellGroup.gahp_')
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)
            
    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Einh':self.Einh,'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_,
                  'g_nmda_':self.g_nmda_,'gahp_':self.gahp_,'tau_ampa':self.tau_ampa,
                  'tau_nmda1':self.tau_nmda1,'tau_nmda2':self.tau_nmda2,
                  'tau_ahp':self.tau_ahp,'eqns':self.eqns}
        return params
    
    def save_parameters(self, out_f):
        for p,v in self.get_parameters().iteritems():
            out_f.write('%s\t%s\n' % (p,str(v)))

def gr_to_go_connections(N_go = 32**2, N_gr = 32**2 * 10**2, dist = 3, p=.05, wrap=False):
    '''
    Implementation of connection matrix from granule cells to golgi cells

    Inspired by T. Yamazaki's code from Yamazaki and Nagao 2012
    from: https://github.com/blennon/Cerebellum/blob/master/okr.c

    N_go: number of Golgi cells
    N_gr: number of Granule cells
    dist: distance in terms of number of Golgi cells (arranged in a grid)
          to consider connecting a cluster of 'n' granule cells to a Golgi
          cell, subject to probability 'p'
    p   : probability of connecting a granule cell cluster to a Golgi cell
    
    Note: This implementation assumes a square grid of neurons.
    
    returns a list of connection tuples [(GR_index,GO_index)]
    '''
    connections = []
    w = int(N_go**.5)
    n = N_gr / N_go
    go_grid = arange(w**2).reshape((w,w))
    
    # iterate over each golgi cell
    for i in range(w):
        for j in range(w):
            
            # get the indices of the surrounding (dist) golgi cells
            if wrap:
                arr_inds = cartesian((arange(i-dist,i+dist+1)%w,arange(j-dist,j+dist+1)%w))
                go_inds = go_grid[arr_inds[:,0],arr_inds[:,1]].reshape((2*dist+1,2*dist+1))
                 
            else:
                go_inds = go_grid[max(i-dist,0):min(i+dist+1,w),max(j-dist,0):min(j+dist+1,w)]
                
            # connect sets of corresponding granule cells to the center
            # golgi cell, selected at random from the set of surrounding golgi cells
            for go_ind in go_inds[rand(go_inds.shape[0],go_inds.shape[1]) < p]:
                for gr_ind in xrange(go_ind*n,go_ind*n + n):
                    connections.append((gr_ind,go_grid[i,j]))
    return list(set(connections))

def go_to_gr_connections(N_go = 32**2, N_gr = 32**2 * 10**2, dist = 4, p=.025, wrap=False):
    '''
    Implementation of connection matrix from golgi cells to granule cells

    Inspired by T. Yamazaki's code from Yamazaki and Nagao 2012
    from: https://github.com/blennon/Cerebellum/blob/master/okr.c

    N_go: number of Golgi cells
    N_gr: number of Granule cells
    dist: distance in terms of number of Golgi cells (arranged in a grid)
          to consider connecting a Golgi cell to a cluster of 'n' granule 
          cells, subject to probability 'p'
    p   : probability of connecting a Golgi cell to a granule cell cluster
          within a window (dist x dist)
    
    Note: This implementation assumes a square grid of neurons.
    
    returns a list of connection tuples [(GO_index,GR_index)]
    '''
    connections = set()
    w = int(N_go**.5)
    n = N_gr / N_go
    go_grid = arange(N_go).reshape((w,w))
    
    # iterate over every golgi cell in a grid of golgi cells, index by (i,j) coordinates
    for i in range(w):
        for j in range(w):
            
            # get the single numeral indices of the surrounding golgi cells
            if wrap:
                go_arr_inds = cartesian((arange(i-dist,i+dist+1)%w,arange(j-dist,j+dist+1)%w))
                go_inds = go_grid[go_arr_inds[:,0],go_arr_inds[:,1]].reshape((2*dist+1,2*dist+1))
            else:
                go_inds = go_grid[max(i-dist,0):min(i+dist+1,w),max(j-dist,0):min(j+dist+1,w)]
                
            # randomly choose a subset of these surrounding golgi cells to connect
            for src_go_ind in go_inds[rand(go_inds.shape[0],go_inds.shape[1]) <= p]:
                
                # connect to glomeruli, i.e. the surrounding four granule cell clusters
                # surrounding is down and to the right
                if wrap:
                    go_gl_arr_inds = cartesian((arange(i,i+2)%w,arange(j,j+2)%w))
                    go_gl_inds = go_grid[go_gl_arr_inds[:,0],go_gl_arr_inds[:,1]]
                else:
                    go_gl_inds = go_grid[i:min(i+2,w),j:min(j+2,w)].flatten()
                    
                for go_gl_ind in go_gl_inds:
                    
                    # connect to all granule cells in a chosen cluster
                    for gr_ind in xrange(go_gl_ind*n, go_gl_ind*n + n):
                        connections.add((src_go_ind,gr_ind))
    return list(connections)
    
        
if __name__ == "__main__":
    from plotting_util import *
    set_global_preferences(useweave=True, usenewpropagate=True, usecodegen=True, usecodegenweave=True)
    
    # Create neuron groups
    N_GO = 32**2
    N_GR = N_GO * 5**2
    N_PG = N_GR
    GO = GolgiCellGroup(N_GO)
    GR = GranuleCellGroup(N_GR)
    PG = PoissonGroup(N_PG, lambda t: 15 * Hz + 30 * Hz * 0.5 * (1 + cos(2 * pi * t * 1.5 * Hz + pi)))
    
    # Create Synapses
    w_gr_go = .2/(49*(len(GR)/len(GO)))
    w_go_gr = 20.
    w_mf_gr = 4.
    S_GO_GR = Synapses(GO,GR,model='w:1',pre='g_inh1+=GR.g_inh_*w_go_gr; g_inh2+=GR.g_inh_*w_go_gr')
    S_GR_GO = Synapses(GR,GO,model='w:1',pre='g_ampa+=GO.g_ampa_*w_gr_go;g_nmda1+=GO.g_ampa_*w_gr_go;g_nmda2+=GO.g_ampa_*w_gr_go')
    S_PG_GR = Synapses(PG,GR,model='w:1',pre='g_ampa+=GR.g_ampa_*w_mf_gr;g_nmda+=GR.g_ampa_*w_mf_gr')
    S_PG_GR.connect_one_to_one()
    for src,trg in gr_to_go_connections(len(GO),len(GR)):
        S_GR_GO[src,trg] = 1.
    for src,trg in go_to_gr_connections(len(GO),len(GR)):
        S_GO_GR[src,trg] = 1.
    
    # Monitor
    MS_PG = SpikeMonitor(PG)
    MS_GR = SpikeMonitor(GR)
    MS_GO = SpikeMonitor(GO)
    
    # Run simulation
    run(2000*ms)
    
    # Plot
    close('all')
    fig1 = figure(1)
    ax = fig1.add_subplot(311)
    plot_raster_firingrate_overlay(MS_PG,range(0,N_GR,N_GR/N_GO),ax)
    title('Poisson Group Raster Plot')
    
    ax = fig1.add_subplot(312)
    plot_raster_firingrate_overlay(MS_GR,range(0,N_GR,N_GR/N_GO),ax)
    title('Granule Cells Raster Plot')
    
    ax = fig1.add_subplot(313)
    plot_raster_firingrate_overlay(MS_GO,range(0,N_GO),ax)
    title('Golgi Cells Raster Plot')
