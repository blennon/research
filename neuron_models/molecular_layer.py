from pylab import *
from brian import *
from abstract_neuron_group import *
from util import *

class MLIGroup(AbstractNeuronGroup):
    '''
    Group of Molecular Layer Interneurons (MLIs).
    
    This model assumes the physiological properties of basket and stellate 
    cells are similar enough to share the same model.
    
    Modeled as conductance-based leaky-integrate-and-fire neurons.
    '''
    
    def __init__(self, 
                 N,
                 Vth = -53. * mvolt,           # Firing threshold, Volts -- Midtgaard (1992)
                 Cm = 14.6 * pfarad,           # Membrane capacitance -- Hausser and Clark (1997)           
                 El = -68. * mvolt,            # leak reversal potential -- derived
                 Eex = 0. * mvolt,             # Excitatory reversal potential -- Carter and Regehr (2002)
                 Einh = -82. * mvolt,          # Inhibitory reversal potential -- Carter and Regehr (2002)
                 Eahp = -82. * mvolt,          # After hyperpolarization reversal potential -- derived to match Lachamp et al. (2009)
                 gl = 1.6 * nsiemens,          # maximum leak conductance -- derived from Hausser and Clark (1997)
                 g_ahp_ = 50. * nsiemens,      # maximum after hyperpolarization conductance 100 used in paper -- derived to match Lachamp et al. (2009)
                 g_inh_ = 4. * nsiemens,       # maximum inhibitory conductance -- Carter and Regehr (2002)
                 g_ampa_ = 3 * nsiemens,       # maximum AMPAR mediated synaptic conductance -- Carter and Regehr (2002)
                 g_nmda_ = 1 * nsiemens,       # maximum NMDAR mediated conductance -- derived from Carter and Regehr (2000)
                 tau_ahp = 2.5 * msecond,      # AHP time constant -- derived to resemble Lachamp et al. (2009)
                 tau_inh = 4.6 * msecond,      # Inhbitory time constant -- Carter and Regehr (2002)
                 tau_ampa_fast = .8 * msecond, # AMPAR unitary EPSC time constant -- Carter and Regehr (2002)
                 tau_ampa_slow = 18 * msecond, # EPSC slow time constant -- Satake, Inoue and Imoto (2012)
                 tau_nmda_rise = 3 * msecond,  # NMDAR rise time constant -- Gabbiani et al. (1994)
                 tau_nmda_decay = 40 * msecond,# NMDAR decay time constant -- Gabbiani et al. (1994)
                 tau_n_decay = 10 * msecond,   # Spillover neurotransmitter decay constant
                 alpha_ampa_fast = .8,         # EPSC fast component -- Satake, Inoue and Imoto (2012)
                 alpha_ampa_slow = .2,         # EPSC slow component -- Satake, Inoue and Imoto (2012)
                 rand_V_init = True,
                 tau_adjust = True,
                 **kwargs):

        self.Vth, self.Cm, self.El, self.Eex, self.Einh, self.Eahp = Vth, Cm, El, Eex, Einh, Eahp
        self.gl, self.g_ampa_, self.g_inh_, self.g_ahp_ = gl, g_ampa_, g_inh_, g_ahp_
        if tau_adjust:
            dt = defaultclock.dt
            tau_ampa_fast = adjust_tau(dt, tau_ampa_fast)
            tau_ampa_slow = adjust_tau(dt, tau_ampa_slow)
            tau_inh = adjust_tau(dt, tau_inh)
            tau_ahp = adjust_tau(dt, tau_ahp)
        self.tau_ampa_fast, self.tau_ampa_slow,  self.tau_inh, self.tau_ahp = tau_ampa_fast, tau_ampa_slow, tau_inh, tau_ahp
        self.alpha_ampa_fast, self.alpha_ampa_slow = alpha_ampa_fast, alpha_ampa_slow
               
        self.eqns = Equations('''
        # Membrane equation
        dV/dt = 1/Cm*(-gl*(V-El)-(g_ampa+gv_nmda)*(V-Eex)-g_ahp*(V-Eahp)-g_inh*(V-Einh) + I) : mV
        
        # After hyperpolarization
        dg_ahp/dt = -g_ahp/tau_ahp : nS
        
        # AMPARs
        dg_ampa_fast/dt = -g_ampa_fast/tau_ampa_fast : nS
        dg_ampa_slow/dt = -g_ampa_slow/tau_ampa_slow : nS
        g_ampa = alpha_ampa_fast*g_ampa_fast + alpha_ampa_slow*g_ampa_slow : nS

        # NMDARs and spillover neurotransmitter (n) availability
        dg_nmda/dt = -g_nmda/tau_nmda_decay + (n**2)*(1-g_nmda)/tau_nmda_rise : 1
        dn/dt = -n/tau_n_decay : 1
        gv_nmda = g_nmda_*g_nmda*(1+exp(-V/(16.13*mV))*(1.2/3.57))**-1 : nS # Voltage gated

        # GABA
        dg_inh/dt = -g_inh/tau_inh : nS
        
        # Input current
        I : nA
        ''')
                
        super(MLIGroup, self).__init__(N, self.eqns,Vth,reset='g_ahp=g_ahp_',**kwargs)
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)
            
        if self.clock.dt > .25*ms:
            warnings.warn('Clock for MLI group should be .25*ms for numerical stability')

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 'g_ahp_':self.g_ahp_, 
                  'tau_ampa':self.tau_ampa, 'tau_ahp':self.tau_ahp,'eqns':self.eqns
                  }
        return params
    
    
class BasketCellGroup(AbstractNeuronGroup):
    '''
    Group of Basket cells
    
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
                 g_ahp_ = 100. * nsiemens,     # maximum after hyperpolarization conductance 100 used in paper
                 g_inh_ = 1. * nsiemens,       # maximum inhibitory conductance
                 tau_ampa = 8.3 * msecond,     # AMPA time constant
                 tau_ahp = 2.5 * msecond,      # AHP time constant
                 tau_inh = 10. * msecond,      # Inhbitory time constant
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
        dV/dt = 1/Cm*(-gl*(V-El)-g_ampa*(V-Eex)-g_ahp*(V-Eahp)-g_inh*(V-Einh) + I) : mV
        
        # After hyperpolarization
        dg_ahp/dt = -g_ahp/tau_ahp : nS
        
        # Glutamate
        dg_ampa/dt = -g_ampa/tau_ampa : nS
        
        # GABA
        dg_inh/dt = -g_inh/tau_inh : nS
        
        # Input current
        I : nA
        ''')
                
        super(BasketCellGroup, self).__init__(N, self.eqns,Vth,reset='g_ahp=g_ahp_', **kwargs)
        
        if rand_V_init:
            self.V = self.El + (self.Vth - self.El)*rand(N)

    def get_parameters(self):
        params = {'N':len(self),'Vth':self.Vth,'Cm':self.Cm,'El':self.El,'Eex':self.Eex,
                  'Eahp':self.Eahp,'gl':self.gl,'g_ampa_':self.g_ampa_, 'g_ahp_':self.g_ahp_, 
                  'tau_ampa':self.tau_ampa, 'tau_ahp':self.tau_ahp,'eqns':self.eqns
                  }
        return params