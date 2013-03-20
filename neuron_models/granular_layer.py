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
    dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_nmda*(V-Eex)-g_inh*(V-Einh)-gahp*(V-Eahp)) : mV
    
    # After hyperpolarization
    dgahp/dt = -gahp/tau_ahp : nS
    
    # Glutamate
    dg_ampa/dt = -g_ampa/tau_ampa : nS
    dg_nmda1/dt = -g_nmda1/tau_nmda1 : nS
    dg_nmda2/dt = -g_nmda1/tau_nmda2 : nS
    g_nmda = .33 * g_nmda1 + .67 * g_nmda2 : nS

    ''')
    
    def __init__(self, N):
        super(GranuleCellGroup, self).__init__(N, model=GolgiCellGroup.eqns,threshold=GolgiCellGroup.Vth,
                                               reset='V=GolgiCellGroup.El;gahp=GolgiCellGroup.gahp_')
        
if __name__ == "__main__":
    GR = GranuleCellGroup(1)
    GO = GolgiCellGroup(1)