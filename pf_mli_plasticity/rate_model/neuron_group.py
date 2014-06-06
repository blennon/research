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
        self.sources, self.weights = [], []
        self.update()

    def connect(self, source, weights):
        '''
        connect source neuron group 'source' with synaptic weights 'weights'
        '''
        self.sources.append(source)
        self.weights.append(weights)

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
        for s,W in zip(self.sources,self.weights):
            self.state += dot(s.get_state(),W)

    def get_state(self):
        '''
        return the state of the neurons
        '''
        return self.state