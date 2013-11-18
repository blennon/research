from pylab import *
from brian import *
from abstract_neuron_group import *
from util import * 

class PurkinjeCellGroup(AbstractNeuronGroup):
    '''
    Group of Purkinje Cells
    
    Note: these are the parameters that match the code published by
    Yamazaki and Nagao 2012
    '''
    
    def __init__(self, 
                 N,
                 Vth = -55. * mvolt,           # Firing threshold, Volts
                 Cm = 106. * pfarad,           # Membrane capacitance
                 El = -68. * mvolt,            # leak reversal potential
                 Eex = 0. * mvolt,             # Excitatory reversal potential
                 Einh = -75. * mvolt,          # Inhibitory reversal potential
                 Eahp = -70. * mvolt,          # After hyperpolarization reversal potential
                 gl = 2.32 * nsiemens,         # maximum leak conductance
                 g_ampa_ = 0.7 * nsiemens,     # maximum ampa conductance
                 g_inh_ = 1. * nsiemens,       # maximum inhibitory conductance
                 g_ahp_ = 100. * nsiemens,     # maximum after hyperpolarization
                 tau_ampa = 8.3 * msecond,     # AMPA time constant
                 tau_inh = 10. * msecond,      # Inhbitory time constant
                 tau_ahp = 2.5 * msecond,      # AHP time constant
                 rand_V_init = True,
                 tau_adjust = True,
                 **kwargs):
        
        self.Vth, self.Cm, self.El, self.Eex, self.Einh, self.Eahp = Vth, Cm, El, Eex, Einh, Eahp
        self.gl, self.g_ampa_, self.g_inh_, self.g_ahp_ = gl, g_ampa_, g_inh_, g_ahp_
        if tau_adjust:
            dt = defaultclock.dt
            tau_ampa = adjust_tau(dt, tau_ampa)
            tau_inh = adjust_tau(dt, tau_inh)
            tau_ahp = adjust_tau(dt, tau_ahp)
        self.tau_ampa,  self.tau_inh, self.tau_ahp = tau_ampa, tau_inh, tau_ahp  

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
        
        super(PurkinjeCellGroup, self).__init__(N,self.eqns,self.Vth,reset='g_ahp=g_ahp_',**kwargs)

        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Einh':self.Einh,'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_,
                  'g_inh_':self.g_inh_,'g_ahp_':self.g_ahp_, 'tau_ampa':self.tau_ampa,
                  'tau_inh':self.tau_inh,'tau_ahp':self.tau_ahp,'eqns':self.eqns,
                  'I_spont':self.I_spont}
        return params


