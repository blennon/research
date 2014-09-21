__author__ = 'bill'
from brian import *

class RealTimeRateMonitor(SpikeMonitor):
    '''
    Converts a spike train for each neuron to a rate code in real time
    using a sum of exponentials kernel.

    The kernel is of the form:
    k(t) = ( exp(-t/tau_fall) - exp(-t/tau_rise) ) / (tau_fall - tau_rise)
    '''
    @check_units(tau_f=ms, tau_r=ms)
    def __init__(self, neuron_group, tau_f=10.*ms, tau_r=2.*ms, record=False):
        '''
        neuron_group: the neuron group for neurons to record
        tau_f: fall time constant
        tau_r: rise time constant
        record: bool, True to store the history of the rate array
        '''
        self.tau_f = tau_f
        self.tau_r = tau_r
        self.record = record
        self.recording = []
        self.f = zeros(len(neuron_group))*Hz
        self.r = zeros(len(neuron_group))*Hz
        super(RealTimeRateMonitor, self).__init__(neuron_group)

    def propagate(self, spikes):
        '''
        update the state of rising and falling firing rate traces

        optionally, record the trace of firing rates
        '''
        dt = defaultclock.dt
        self.f -= self.f*dt/self.tau_f
        self.r -= self.r*dt/self.tau_r
        if len(spikes):
            self.f += 1*Hz
            self.r += 1*Hz
        if record:
            self.recording.append(self.get_firing_rates())

    def get_firing_rates(self):
        '''
        return the neuron firing rates for the current time step
        '''
        return (self.f - self.r)/(self.tau_f - self.tau_r)

    def getvalues(self):
        '''
        return the values of the recording
        '''
        return self.recording