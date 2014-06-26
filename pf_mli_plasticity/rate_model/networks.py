__author__ = 'bill'
from rate_model import *

class Network(object):
    '''
    defines a network object
    '''

    def update(self):
        '''
        update the state of the network
        '''
        raise NotImplementedError

class PF_MLI_Network(Network):

    def __init__(self, N_PF, N_MLI, N_CF, MLI_rest, W_PF_MLI, W_CF_MLI, beta=.001, dt=.1, tau_trace=0.):
        '''
        N_PF: number of PFs
        N_MLI: number of MLIs
        N_CF: number of CFs
        MLI_rest: array, resting state of MLIs
        W_PF_MLI: weight matrix for PF-MLI synapses
        W_CF_MLI: weight matrix for CF-MLI synapses
        beta: learning rate
        dt: time step
        tau_trace: time constant for PF trace in learning rule
        '''

        self.beta = beta

        # define neuron groups
        self.PF = NeuronGroup(N_PF)
        self.PF_trace = PFTrace(N_PF, dt, tau_trace)
        self.MLI = ShuntingNeuronGroup(N_MLI, A=1.5, resting_state=MLI_rest,tau=1.,dt=dt)
        self.CF = NeuronGroup(N_CF)

        # connect neurons
        self.C_PF_MLI = PF_MLI_Connection(self.PF,self.MLI,self.PF_trace,W=W_PF_MLI)
        self.C_CF_MLI = Connection(self.CF,self.MLI,W=W_CF_MLI)
        self.MLI.connect(self.C_PF_MLI)
        self.MLI.connect(self.C_CF_MLI)

        # monitor neuron & weight states
        self.PF_monitor = StateMonitor(self.PF)
        self.PF_trace_monitor = StateMonitor(self.PF_trace)
        self.MLI_monitor = StateMonitor(self.MLI)
        self.CF_monitor = StateMonitor(self.CF)
        self.W_monitor = StateMonitor(self.C_PF_MLI)

    def cf_active(self):
        '''
        returns True if the CF is firing
        '''
        CF = self.CF.get_state()
        if CF[0] > 0:
            return True
        return False

    def update(self):
        '''
        Update the state of the network
        '''
        self.PF_trace.update(self.PF.get_state())
        self.MLI.update()
        self.C_PF_MLI.update(beta=self.beta, CF_active=self.cf_active())

    def record(self):
        self.PF_monitor.record()
        self.PF_trace_monitor.record()
        self.MLI_monitor.record()
        self.CF_monitor.record()
        self.W_monitor.record()

    def reset(self):
        '''
        reset the state of the neuron groups and clear monitors
        '''
        self.PF.reset_state()
        self.MLI.reset_state()
        self.PF_monitor.clear_recording()
        self.PF_trace_monitor.clear_recording()
        self.MLI_monitor.clear_recording()
        self.W_monitor.clear_recording()
        self.CF_monitor.clear_recording()

    def get_monitor_states(self):
        '''
        return recordings from monitors
        '''
        return {'PF':self.PF_monitor.get_recording(),'PF Trace': self.PF_trace_monitor.get_recording(),
                'MLI':self.MLI_monitor.get_recording(),'W':self.W_monitor.get_recording(),
                'CF':self.CF_monitor.get_recording()}