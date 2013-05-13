from pylab import *
from brian import *
from abstract_neuron_group import *

class PurkinjeCellGroup(AbstractNeuronGroup):
    '''
    Group of Purkinje Cells
    
    Note: these are the parameters that match the code published by
    Yamazaki and Nagao 2012
    '''
    Vth = -55. * mvolt           # Firing threshold, Volts
    Cm = 106. * pfarad           # Membrane capacitance

    El = -68. * mvolt            # leak reversal potential
    Eex = 0. * mvolt             # Excitatory reversal potential
    Einh = -75. * mvolt          # Inhibitory reversal potential
    Eahp = -70. * mvolt          # After hyperpolarization reversal potential

    gl = 2.32 * nsiemens         # maximum leak conductance
    g_ampa_ = 0.7 * nsiemens     # maximum ampa conductance
    g_inh_ = 1. * nsiemens       # maximum inhibitory conductance
    g_ahp_ = 100. * nsiemens     # maximum after hyperpolarization
    
    tau_ampa = 8.3 * msecond     # AMPA time constant
    tau_inh = 10. * msecond      # Inhbitory time constant
    tau_ahp = 2.5 * msecond      # AHP time constant
    
    I_spont = 0.25 * nA          # Spontaneous current
    
    def __init__(self, N, rand_V_init = True):
        
        Cm, El, Eex, Einh, Eahp = self.Cm, self.El, self.Eex, self.Einh, self.Eahp
        gl, tau_ampa, tau_inh, tau_ahp = self.gl, self.tau_ampa, self.tau_inh, self.tau_ahp

        self.eqns = Equations('''
        # Membrane equation
        dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_inh*(V-Einh)-g_ahp*(V-Eahp) + I) : mV
        
        # After hyperpolarization
        dg_ahp/dt = -g_ahp/tau_ahp : nS
        
        # Glutamate
        dg_ampa/dt = -g_ampa/tau_ampa : nS
        
        # GABA
        dg_inh/dt = -g_inh/tau_inh : nS
        
        # Input current
        I : nA
        ''')
        
        super(PurkinjeCellGroup, self).__init__(N, model=self.eqns,threshold=self.Vth,
                                               reset='g_ahp=self.g_ahp_') # Yamazaki and Nagao 2012 don't reset voltage
        self.I = self.I_spont
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Einh':self.Einh,'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_,
                  'g_inh_':self.g_inh_,'g_ahp_':self.g_ahp_, 'tau_ampa':self.tau_ampa,
                  'tau_inh':self.tau_inh,'tau_ahp':self.tau_ahp,'eqns':self.eqns,
                  'I_spont':self.I_spont}
        return params


