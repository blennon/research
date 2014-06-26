__author__ = 'bill'
from pylab import *

class NeuronGroup(object):
    '''
    Defines a rate based model for a neuron group with 'N' neurons
    '''

    def __init__(self, N, resting_state=None, name=None):
        '''
        N: number of neurons in this group
        '''
        self.N = N
        self.state = zeros(N)
        self.resting_state = resting_state
        self.name = name
        self.connections = []
        self.reset_state()

    def connect(self, connection):
        '''
        connect this group with a connection object
        '''
        self.connections.append(connection)

    def set_state(self, state):
        '''
        set the state of the neurons to 'state'
        '''
        if state.shape != self.state.shape:
            raise Exception('Mismatch between neuron group dimension and input state dimension')
        self.state = state

    def reset_state(self):
        '''
        reset the state of the neurons
        '''
        self.set_state(zeros(self.N))
        if self.resting_state is not None:
            self.set_state(self.resting_state.copy())

    def update(self):
        '''
        update the state of the neurons given the inputs
        '''
        self.reset_state()
        for C in self.connections:
            polarity = C.get_synapse_polarity()
            self.state += polarity * dot(C.get_state(), C.get_source_state())

    def get_state(self):
        '''
        return the state of the neurons
        '''
        return self.state

    def get_num_neurons(self):
        '''
        return the number of neurons in the neuron group
        '''
        return self.state.shape[0]

    def get_connection(self, name):
        '''
        returns the connection
        '''
        return self.connections


class ShuntingNeuronGroup(NeuronGroup):
    '''
    Implements shunting neuron dynamics

    du/dt = -A(u-B) + (C-Du)I_exc - (E+Fu)I_inh
    where B is the resting state
    '''

    def __init__(self, N, resting_state, A=1, C=1, D=1, E=0, F=1, tau=.5, dt=.1):
        super(ShuntingNeuronGroup, self).__init__(N, resting_state)
        self.A, self.C, self.D, self.E, self.F = A,C,D,E,F
        self.tau = tau
        self.dt = dt

    def update(self):
        '''
        update the state of the network
        '''
        I_exc = self.synaptic_input('excitatory')
        I_inh = self.synaptic_input('inhibitory')
        dudt = -self.A*(self.state - self.resting_state) + (self.C-self.D*self.state)*I_exc - \
               (self.E+self.F*self.state)*I_inh
        self.state += self.dt*self.tau*dudt

    def synaptic_input(self, polarity):
        '''
        compute the total synaptic input across all connections of a certain polarity, i.e.
        excitatory or inhibitory
        '''
        if polarity == 'excitatory':
            sign = 1
        elif polarity == 'inhibitory':
            sign = -1
        else:
            raise Exception('polarity must be either excitatory or inhibitory')

        I = 0.
        for C in self.connections:
            if C.get_synapse_polarity() == sign:
                I += dot(C.get_state(), C.get_source_state())
        return I

class PFTrace(NeuronGroup):
    '''
    Keeps a trace of the PF activity
    '''

    def __init__(self, N, dt=.1, tau=0.):
        super(PFTrace, self).__init__(N, resting_state=zeros(N), name='PF trace')
        self.dt = dt
        self.tau = tau

    def update(self, PF_state):
        '''
        updates the state of the PF trace

        tau: trace time constant, if zero the trace is the instantaneous value of the input
        '''
        if self.tau > 0:
            dP_dt = (1./self.tau)*(PF_state - self.state)
            self.state += self.dt*dP_dt
        else:
            self.state = PF_state