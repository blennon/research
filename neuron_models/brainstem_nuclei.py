from pylab import *
from brian import *
from abstract_neuron_group import *
from util import *

class InferiorOliveGroup(AbstractNeuronGroup):
    '''
    Group of cells from the inferior olive
    
    Note: these are the parameters that match the code published by
    Yamazaki and Nagao 2012
    '''
     
    def __init__(self, 
                 N,
                 Vth = -50. * mvolt,          # Firing threshold, Volts
                 Cm = 1. * pfarad,           # Membrane capacitance
                 El = -60. * mvolt,           # leak reversal potential
                 Eex = 0. * mvolt,            # Excitatory reversal potential
                 Einh = -75. * mvolt,         # Inhibitor reversal potential   
                 Eahp = -70. * mvolt,         # After hyperpolarization reversal potential            
                 gl = 0.015 * nsiemens,       # maximum leak conductance
                 g_ampa_ = .1 * nsiemens,     # maximum AMPA conductance
                 g_inh_ = 0.018 * nsiemens,  # maximum inhibitor conductance
                 g_ahp_ = 1. * nsiemens,      # maximum after hyperpolarization conductance 100 used in paper                
                 tau_ampa = 10. * msecond,    # AMPA time constant
                 tau_inh = 10. * msecond,    # GABA time constant
                 tau_ahp = 5. * msecond,      # AHP time constant 
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
        dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_inh*(V-Einh)-g_ahp*(V-Eahp)) : mV
        
        # After hyperpolarization
        dg_ahp/dt = -g_ahp/tau_ahp : nS
        
        # Glutamate
        dg_ampa/dt = -g_ampa/tau_ampa : nS
        
        # GABA
        dg_inh/dt = -g_inh/tau_inh : nS
        ''')
        
        super(InferiorOliveGroup, self).__init__(N, self.eqns,threshold=Vth,reset='V=self.El;g_ahp=self.g_ahp_', **kwargs)
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Einh':self.Einh,'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 
                  'g_inh_':self.g_inh_, 'g_ahp_':self.g_ahp_, 'tau_ampa':self.tau_ampa, 
                  'tau_inh':self.tau_inh,'tau_ahp':self.tau_ahp,'eqns':self.eqns
                  }
        return params

class VestibularNucleusGroup(AbstractNeuronGroup):
    '''
    Group of cells from the vestibular nucleus
    
    Note: these are the parameters that match the code published by
    Yamazaki and Nagao 2012
    '''
    
    def __init__(self, 
                 N,
                 Vth = -38.8 * mvolt,          # Firing threshold, Volts
                 Cm = 122.3 * pfarad,          # Membrane capacitance
                 El = -56. * mvolt,            # leak reversal potential
                 Eex = 0. * mvolt,             # Excitatory reversal potential
                 Einh = -88. * mvolt,          # Inhibitor reversal potential   
                 Eahp = -70. * mvolt,          # After hyperpolarization reversal potential
                 gl = 1./.61 * nsiemens,       # maximum leak conductance
                 g_ampa_ = 50. * nsiemens,     # maximum AMPA conductance
                 g_nmda_ = 25.8 * nsiemens,    # maximum NMDA conductance
                 g_inh_ = 30. * nsiemens,      # maximum inhibitor conductance
                 g_ahp_ = 50. * nsiemens,      # maximum after hyperpolarization conductance 100 used in paper
                 tau_ampa = 9.9 * msecond,     # AMPA time constant
                 tau_nmda = 30.5 * msecond,    # NMDA time constant
                 tau_inh = 42.3 * msecond,     # GABA time constant
                 tau_ahp = 5. * msecond,       # AHP time constant
                 I_spont = .7 * nA,            # Intrinsic current 
                 rand_V_init = True,
                 tau_adjust = True, 
                 **kwargs):
        
        self.Vth, self.Cm, self.El, self.Eex, self.Einh, self.Eahp = Vth, Cm, El, Eex, Einh, Eahp
        self.gl, self.g_ampa_, self.g_nmda_, self.g_inh_, self.g_ahp_ = gl, g_ampa_, g_nmda_, g_inh_, g_ahp_
        if tau_adjust:
            dt = defaultclock.dt
            tau_ampa = adjust_tau(dt, tau_ampa)
            tau_nmda = adjust_tau(dt, tau_nmda)
            tau_inh = adjust_tau(dt, tau_inh)
            tau_ahp = adjust_tau(dt, tau_ahp)
        self.tau_ampa, self.tau_nmda, self.tau_inh, self.tau_ahp = tau_ampa, tau_nmda, tau_inh, tau_ahp    
        
        self.eqns = Equations('''
        # Membrane equation
        dV/dt = 1/Cm*(-gl*(V-El)-g_ex*(V-Eex)-g_inh*(V-Einh)-g_ahp*(V-Eahp) + I_spont) : mV
        
        # After hyperpolarization
        dg_ahp/dt = -g_ahp/tau_ahp : nS
        
        # Glutamate
        dg_ampa/dt = -g_ampa/tau_ampa : nS
        dg_nmda/dt = -g_nmda/tau_nmda : nS
        g_ex = .66*g_ampa + .34*g_nmda :nS
        
        # GABA
        dg_inh/dt = -g_inh/tau_inh : nS
        ''')
        
        super(VestibularNucleusGroup, self).__init__(N, self.eqns,self.Vth,reset='g_ahp=self.g_ahp_', **kwargs)
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Einh':self.Einh,'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 
                  'g_nmda_':self.g_nmda_, 'g_inh_':self.g_inh_, 'g_ahp_':self.g_ahp_, 
                  'tau_ampa':self.tau_ampa, 'tau_nmda':self.tau_nmda, 'tau_inh':self.tau_inh,
                  'tau_ahp':self.tau_ahp,'I_spont':self.I_spont, 'r_ampa':self.r_ampa, 
                  'r_nmda':self.r_nmda,'eqns':self.eqns
                  }
        return params