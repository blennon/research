__author__ = 'bill'
from pylab import *

class StateMonitor(object):
    '''
    record the state of the neuron group
    '''
    def __init__(self, neuron_group):
        self.neuron_group = neuron_group
        self.history = []
        self.record()

    def record(self):
        self.history.append(self.neuron_group.get_state().copy())

    def get_recording(self):
        return squeeze(self.history).T

    def clear_recording(self):
        self.history = []