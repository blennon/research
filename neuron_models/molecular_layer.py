from pylab import *
from brian import *
from abstract_neuron_group import *

class BasketCellGroup(AbstractNeuronGroup):
    '''
    Group of Basket cells
    
    Note: these are the parameters that match the code published by
    Yamazaki and Nagao 2012
    '''
    Vth = -55. * mvolt           # Firing threshold, Volts
    Cm = 106. * pfarad           # Membrane capacitance

    El = -68. * mvolt            # leak reversal potential
    Eex = 0. * mvolt             # Excitatory reversal potential
    Eahp = -70. * mvolt          # After hyperpolarization reversal potential

    gl = 2.32 * nsiemens         # maximum leak conductance
    g_ampa_ = 0.7 * nsiemens     # maximum ampa conductance
    g_ahp_ = 100. * nsiemens       # maximum after hyperpolarization conductance 100 used in paper
    
    tau_ampa = 8.3 * msecond     # AMPA time constant
    tau_ahp = 2.5 * msecond      # AHP time constant
    
    eqns = Equations('''
    # Membrane equation
    dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_ahp*(V-Eahp)) : mV
    
    # After hyperpolarization
    dg_ahp/dt = -g_ahp/tau_ahp : nS
    
    # Glutamate
    dg_ampa/dt = -g_ampa/tau_ampa : nS
    ''')
    
    def __init__(self, N, rand_V_init = True):
        
        super(BasketCellGroup, self).__init__(N, model=BasketCellGroup.eqns,threshold=BasketCellGroup.Vth,
                                               reset='g_ahp=BasketCellGroup.g_ahp_') # Yamazaki and Nagao 2012 don't reset voltage
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 'g_ahp_':self.g_ahp_, 
                  'tau_ampa':self.tau_ampa, 'tau_ahp':self.tau_ahp,'eqns':self.eqns
                  }
        return params