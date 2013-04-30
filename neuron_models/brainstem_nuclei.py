from pylab import *
from brian import *
from abstract_neuron_group import *

class InferiorOliveGroup(AbstractNeuronGroup):
    '''
    Group of cells from the inferior olive
    
    Note: these are the parameters that match the code published by
    Yamazaki and Nagao 2012
    '''
    Vth = -50. * mvolt          # Firing threshold, Volts
    Cm = 1. * pfarad            # Membrane capacitance

    El = -60. * mvolt           # leak reversal potential
    Eex = 0. * mvolt            # Excitatory reversal potential
    Einh = -75. * mvolt         # Inhibitor reversal potential   
    Eahp = -70. * mvolt         # After hyperpolarization reversal potential

    gl = 0.015 * nsiemens       # maximum leak conductance
    g_ampa_ = .1 * nsiemens     # maximum AMPA conductance
    g_gaba_ = 0.018 * nsiemens  # maximum inhibitor conductance
    g_ahp_ = 1. * nsiemens      # maximum after hyperpolarization conductance 100 used in paper
    
    tau_ampa = 10. * msecond    # AMPA time constant
    tau_gaba = 10. * msecond    # GABA time constant
    tau_ahp = 5. * msecond      # AHP time constant
    
    eqns = Equations('''
    # Membrane equation
    dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_gaba*(V-Einh)-g_ahp*(V-Eahp)) : mV
    
    # After hyperpolarization
    dg_ahp/dt = -g_ahp/tau_ahp : nS
    
    # Glutamate
    dg_ampa/dt = -g_ampa/tau_ampa : nS
    
    # GABA
    dg_gaba/dt = -g_gaba/tau_gaba : nS
    ''')
    
    def __init__(self, N, rand_V_init = True, **kwargs):
        
        super(InferiorOliveGroup, self).__init__(N, model=self.eqns,threshold=self.Vth,
                                               reset='V=self.El;g_ahp=self.g_ahp_', **kwargs)
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Einh':self.Einh,'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 
                  'g_gaba_':self.g_gaba_, 'g_ahp_':self.g_ahp_, 'tau_ampa':self.tau_ampa, 
                  'tau_gaba':self.tau_gaba,'tau_ahp':self.tau_ahp,'eqns':self.eqns
                  }
        return params

class VestibularNucleusGroup(AbstractNeuronGroup):
    '''
    Group of cells from the vestibular nucleus
    
    Note: these are the parameters that match the code published by
    Yamazaki and Nagao 2012
    '''
    Vth = -38.8 * mvolt          # Firing threshold, Volts
    Cm = 122.3 * pfarad          # Membrane capacitance

    El = -56. * mvolt            # leak reversal potential
    Eex = 0. * mvolt             # Excitatory reversal potential
    Einh = -88. * mvolt          # Inhibitor reversal potential   
    Eahp = -70. * mvolt          # After hyperpolarization reversal potential

    gl = 1./.61 * nsiemens       # maximum leak conductance
    g_ampa_ = 50. * nsiemens     # maximum AMPA conductance
    g_nmda_ = 25.8 * nsiemens    # maximum NMDA conductance
    g_gaba_ = 30. * nsiemens      # maximum inhibitor conductance
    g_ahp_ = 50. * nsiemens      # maximum after hyperpolarization conductance 100 used in paper
    
    tau_ampa = 9.9 * msecond     # AMPA time constant
    tau_nmda = 30.5 * msecond    # NMDA time constant
    tau_gaba = 42.3 * msecond    # GABA time constant
    tau_ahp = 5. * msecond       # AHP time constant
    
    I_spont = .7 * nA            # Intrinsic current
    
    r_ampa = .66                 # proportion of AMPARs
    r_nmda = 1 - r_ampa          # propoertion of NMDARs
    
    eqns = Equations('''
    # Membrane equation
    dV/dt = 1/Cm*(-gl*(V-El)-g_ex*(V-Eex)-g_gaba*(V-Einh)-g_ahp*(V-Eahp) + I_spont) : mV
    
    # After hyperpolarization
    dg_ahp/dt = -g_ahp/tau_ahp : nS
    
    # Glutamate
    dg_ampa/dt = -g_ampa/tau_ampa : nS
    dg_nmda/dt = -g_nmda/tau_nmda : nS
    g_ex = r_ampa*g_ampa + r_nmda*g_nmda :nS
    
    # GABA
    dg_gaba/dt = -g_gaba/tau_gaba : nS
    ''')
    
    def __init__(self, N, rand_V_init = True, **kwargs):
        
        super(VestibularNucleusGroup, self).__init__(N, model=self.eqns,threshold=self.Vth,
                                               reset='V=self.El;g_ahp=self.g_ahp_', **kwargs)
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Einh':self.Einh,'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 
                  'g_nmda_':self.g_nmda_, 'g_gaba_':self.g_gaba_, 'g_ahp_':self.g_ahp_, 
                  'tau_ampa':self.tau_ampa, 'tau_nmda':self.tau_nmda, 'tau_gaba':self.tau_gaba,
                  'tau_ahp':self.tau_ahp,'I_spont':self.I_spont, 'r_ampa':self.r_ampa, 
                  'r_nmda':self.r_nmda,'eqns':self.eqns
                  }
        return params