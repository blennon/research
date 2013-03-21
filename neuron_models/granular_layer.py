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


def gr_to_go_connections(N_go = 32**2, N_gr = 32**2 * 10**2, dist = 3, p=.05, wrap=False):
    '''
    Implementation of connection matrix from granule cells to golgi cells

    Direct copy from T. Yamazaki's code from Yamazaki and Nagao 2012
    from: https://github.com/blennon/Cerebellum/blob/master/okr.c

    N_go: number of Golgi cells
    N_gr: number of Granule cells
    dist: distance in terms of number of Golgi cells (arranged in a grid)
          to consider connecting a cluster of 'n' granule cells to a Golgi
          cell, subject to probability 'p'
    p   : probability of connecting a granule cell cluster to a Golgi cell
    wrap: if true, this implements Yamazaki's code where connections can be
          made from one side of the board to another (pac man style)
    
    Note: This implementation assumes a square grid of neurons.
    
    returns a list of connection tuples [(GR_index,GO_index)]
    '''
    gr_go_connections = []
    w = int(N_go**.5)
    n = N_gr / N_go
    for go_x in range(w):
        for go_y in range(w):
            go_n = go_y + (w)*go_x
            for go_dx in range(-dist,dist+1):
                go_ax = go_x + go_dx
                if wrap:
                    if go_ax >= w: go_ax -= w
                    if go_ax < 0: go_ax += w
                else:
                    if go_ax >= w or go_ax < 0: continue
                for go_dy in range(-dist,dist+1):
                    go_ay = go_y + go_dy
                    if wrap:
                        if go_ay >= w: go_ay -= w
                        if go_ay < 0: go_ay += w
                    else:
                        if go_ay >= w or go_ay < 0: continue
                    go_an = go_ay + (w)*go_ax
                    if rand() < p:
                        for i in range(n):
                            gr_go_connections.append((i+n*go_an,go_n))
    return gr_go_connections

def go_to_gr_connections(N_go = 32**2, N_gr = 32**2 * 10**2, dist = 4, p=.025):
    pass
    
        
if __name__ == "__main__":
    GR = GranuleCellGroup(1)
    GO = GolgiCellGroup(1)