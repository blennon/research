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
        rec = squeeze(self.history).T
        if len(rec.shape) == 1:
            return rec[None,...]
        return rec

    def clear_recording(self):
        self.history = []

    def plot(self, ind=0, **plot_params):
        '''
        plot the state of neuron number 'ind'
        '''
        plot(self.get_recording()[ind,:], **plot_params)