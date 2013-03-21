from pylab import *
from brian import *

class GranuleCellGroup(NeuronGroup):
    '''
    Group of granule cells
    
    Notes: I'm not sure this is the most efficient way to simulate these neurons in BRIAN
    '''
    Vth = -35 * mvolt            # Firing threshold, Volts
    Cm = 3.1 * pfarad            # Membrane capacitance

    El = -58 * mvolt             # leak reversal potential
    Eex = 0. * mvolt             # Excitatory reversal potential
    Einh = -82. * mvolt          # Inhibitory reversal potential
    Eahp = -82. * mvolt          # After hyperpolarization reversal potential

    gl = 0.42 * nsiemens         # maximum leak conductance
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
    dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_nmda*(V-Eex)-g_inh*(V-Einh)-gahp*(V-Eahp)) : mV
    
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
    
    def __init__(self, N):
        
        super(GranuleCellGroup, self).__init__(N, model=GranuleCellGroup.eqns,threshold=GranuleCellGroup.Vth,
                                               reset='V=GranuleCellGroup.El;gahp=GranuleCellGroup.gahp_')
        
class GolgiCellGroup(NeuronGroup):
    '''
    Group of Golgi cells
    '''
    Vth = -52. * mvolt            # Firing threshold, Volts
    Cm = 28. * pfarad            # Membrane capacitance

    El = -55. * mvolt             # leak reversal potential
    Eex = 0. * mvolt             # Excitatory reversal potential
    Einh = -82. * mvolt          # Inhibitory reversal potential
    Eahp = -72.7 * mvolt          # After hyperpolarization reversal potential

    gl = 2.3 * nsiemens         # maximum leak conductance
    g_ampa_ = 45.5 * nsiemens    # maximum ampa conductance
    g_nmda_ = 30. * nsiemens   # maximum nmda conductance
    gahp_ = 20. * nsiemens        # maximum after hyperpolarization conductance
    
    tau_ampa = 1.5 * msecond     # AMPA time constant
    tau_nmda1 = 31. * msecond     # NMDA time constant 1
    tau_nmda2 = 170. * msecond     # NMDA time constant 2    
    tau_ahp = 5. * msecond       # AHP time constant
    
    eqns = Equations('''
    # Membrane equation
    dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_nmda*(V-Eex)-gahp*(V-Eahp)) : mV
    
    # After hyperpolarization
    dgahp/dt = -gahp/tau_ahp : nS
    
    # Glutamate
    dg_ampa/dt = -g_ampa/tau_ampa : nS
    dg_nmda1/dt = -g_nmda1/tau_nmda1 : nS
    dg_nmda2/dt = -g_nmda1/tau_nmda2 : nS
    g_nmda = .33 * g_nmda1 + .67 * g_nmda2 : nS

    ''')
    
    def __init__(self, N):
        super(GolgiCellGroup, self).__init__(N, model=GolgiCellGroup.eqns,threshold=GolgiCellGroup.Vth,
                                               reset='V=GolgiCellGroup.El;gahp=GolgiCellGroup.gahp_')


def gr_to_go_connections(N_go = 32**2, N_gr = 32**2 * 10**2, dist = 3, p=.05):
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
    C = arange(w**2).reshape((w,w))
    for i in range(w):
        for j in range(w):
            go_inds = C[max(i-dist,0):min(i+dist+1,w),max(j-dist,0):min(j+dist+1,w)]
            for go_ind in go_inds[rand(go_inds.shape[0],go_inds.shape[1]) < p]:
                for gr_ind in xrange(go_ind*n,go_ind*n + n):
                    connections.append((gr_ind,C[i,j]))
    return connections

def go_to_gr_connections(N_go = 32**2, N_gr = 32**2 * 10**2, dist = 3, p=.025):
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
            go_inds = go_grid[max(i-dist,0):min(i+dist+1,w),max(j-dist,0):min(j+dist+1,w)]
            
            # randomly choose a subset of these surrounding golgi cells to connect
            for src_go_ind in go_inds[rand(go_inds.shape[0],go_inds.shape[1]) <= p]:
                
                # connect to glomeruli, i.e. the surrounding four granule cell clusters
                for go_gl_ind in go_grid[i:min(i+2,w),j:min(j+2,w)].flatten():
                    
                    # connect to all granule cells in a chosen cluster
                    for gr_ind in xrange(go_gl_ind*n, go_gl_ind*n + n):
                        connections.add((src_go_ind,gr_ind))
    return list(connections)
    
        
if __name__ == "__main__":
    GR = GranuleCellGroup(1)
    GO = GolgiCellGroup(1)