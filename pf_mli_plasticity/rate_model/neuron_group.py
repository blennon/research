__author__ = 'bill'
from pylab import *

class NeuronGroup(object):
    '''
    Defines a rate based model for a neuron group with 'N' neurons
    '''

    def __init__(self, N, resting_state=None):
        '''
        N: number of neurons in this group
        '''
        self.N = N
        self.state = zeros(N)
        self.resting_state = resting_state
        self.connections = []
        self.update()

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