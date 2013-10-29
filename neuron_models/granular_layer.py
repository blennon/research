from pylab import *
from brian import *
from abstract_neuron_group import AbstractNeuronGroup
from util import *

class GranuleCellGroup(AbstractNeuronGroup):
    '''
    Group of granule cells
    '''
    
    def __init__(self, 
                 N,
                 Vth = -35 * mvolt,            # Firing threshold, Volts
                 Cm = 3.1 * pfarad,            # Membrane capacitance
                 El = -58 * mvolt,             # leak reversal potential
                 Eex = 0. * mvolt,             # Excitatory reversal potential
                 Einh = -82. * mvolt,          # Inhibitory reversal potential
                 Eahp = -82. * mvolt,          # After hyperpolarization reversal potential
                 gl = 0.43 * nsiemens,         # maximum leak conductance
                 g_ampa_ = 0.18 * nsiemens,    # maximum ampa conductance
                 g_nmda_ = 0.025 * nsiemens,   # maximum nmda conductance
                 g_inh_ = 0.028 * nsiemens,    # maximum inhibitory conductance
                 gahp_ = 1. * nsiemens,        # maximum after hyperpolarization conductance
                 tau_ampa = 1.2 * msecond,     # AMPA time constant
                 tau_nmda = 52. * msecond,     # NMDA time constant
                 tau_inh1 = 7. * msecond,      # Inhbitory time constant 1
                 tau_inh2 = 59. * msecond,     # Inhbitory time constant 2
                 tau_ahp = 5. * msecond,       # AHP time constant 
                 rand_V_init = True,
                 tau_adjust = True, 
                 **kwargs):
        
        self.Vth, self.Cm, self.El, self.Eex, self.Einh, self.Eahp = Vth, Cm, El, Eex, Einh, Eahp
        self.gl, self.g_ampa_, self.g_nmda_, self.g_inh_, self.gahp_ = gl, g_ampa_, g_nmda_, g_inh_, gahp_
        if tau_adjust:
            dt = defaultclock.dt
            tau_ampa = adjust_tau(dt, tau_ampa)
            tau_nmda = adjust_tau(dt, tau_nmda)
            tau_inh1 = adjust_tau(dt, tau_inh1)
            tau_inh2 = adjust_tau(dt, tau_inh2)
            tau_ahp = adjust_tau(dt, tau_ahp)
        self.tau_ampa, self.tau_nmda, self.tau_inh1, self.tau_inh2 = tau_ampa, tau_nmda, tau_inh1, tau_inh2
        self.tau_ahp = tau_ahp        

        self.eqns = Equations('''
        # Membrane equation
        dV/dt = 1/Cm*(-gl*(V-El)-g_ex*(V-Eex)-g_inh*(V-Einh)-gahp*(V-Eahp)) : mV
        
        # After hyperpolarization
        dgahp/dt = -gahp/tau_ahp : nS
        
        # Glutamate
        dg_ampa/dt = -g_ampa/tau_ampa : nS
        dg_nmda/dt = -g_nmda/tau_nmda : nS
        g_ex = .88*g_ampa + .12*g_nmda : nS
        
        # GABA
        dg_inh1/dt = -g_inh1/tau_inh1 : nS
        dg_inh2/dt = -g_inh2/tau_inh2 : nS
        g_inh = .43 * g_inh1 + .57 * g_inh2 : nS
        ''')
        
        super(GranuleCellGroup, self).__init__(N, self.eqns,threshold=Vth,reset='V=El;gahp=gahp_', **kwargs)

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
    
    def __init__(self, 
                 N, 
                 Vth = -52*mvolt,
                 Cm = 28. * pfarad,           # Membrane capacitance
                 El = -55. * mvolt,           # leak reversal potential
                 Eex = 0. * mvolt,            # Excitatory reversal potential
                 Einh = -82. * mvolt,         # Inhibitory reversal potential
                 Eahp = -72.7 * mvolt,        # After hyperpolarization reversal potential
                 gl = 2.3 * nsiemens,         # maximum leak conductance
                 g_ampa_ = 45.5 * nsiemens,   # maximum ampa conductance
                 g_nmda_ = 30. * nsiemens,    # maximum nmda conductance
                 gahp_ = 20. * nsiemens,      # maximum after hyperpolarization conductance
                 tau_ampa = 1.5 * msecond,    # AMPA time constant
                 tau_nmda1 = 31. * msecond,   # NMDA time constant 1
                 tau_nmda2 = 170. * msecond,  # NMDA time constant 2    
                 tau_ahp = 5. * msecond,      # AHP time constant 
                 rand_V_init = True,          # Randomly initialize voltage
                 tau_adjust = True,            # If true, adjust the time constants for equivalence
                 **kwargs):

        self.Vth, self.Cm, self.El, self.Eex, self.Einh, self.Eahp = Vth, Cm, El, Eex, Einh, Eahp
        self.gl, self.g_ampa_, self.g_nmda_, self.gahp_ = gl, g_ampa_, g_nmda_, gahp_
        if tau_adjust:
            dt = defaultclock.dt
            tau_ampa = adjust_tau(dt, tau_ampa)
            tau_nmda1 = adjust_tau(dt, tau_nmda1)
            tau_nmda2 = adjust_tau(dt, tau_nmda2)
            tau_ahp = adjust_tau(dt, tau_ahp)
        self.tau_ampa, self.tau_nmda1, self.tau_nmda2, self.tau_ahp = tau_ampa, tau_nmda1, tau_nmda2, tau_ahp          
         
        self.eqns = Equations('''
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
        
        super(GolgiCellGroup, self).__init__(N, self.eqns, threshold=Vth, reset='V=El;gahp=gahp_', **kwargs)
        if rand_V_init:
            self.V = self.El + (Vth - El)*rand(N)
            
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
