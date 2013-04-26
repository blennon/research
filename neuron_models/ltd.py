from pylab import *
from brian import *

class LTD(SpikeMonitor):
    '''
    Depress PF-PKJ synapses for GRs that spiked that synapse on
    PKJs that received CF spike.
    '''
    def __init__(self, IO, GR, PF_PKJ, CF_PKJ, ltd_decay=.995, window=50*ms):
        '''
        IO: NeuronGroup of inferior olive neurons
        GR: NeuronGroup of granule cells
        PF_PKJ: Synapses object of PF-PKJ synapses
        CF_PKJ: Synapses object of CF-PKJ synapses
        ltd_decay: the multiplicative decay constant to decay PF-PKJ
                   synapses by.
        window: the time window to depress PF-PKJ synapses for GR spikes 
                that precede IO spikes.
        '''
        self.IO = IO
        self.GR = GR
        self.PF_PKJ = PF_PKJ
        self.CF_PKJ = CF_PKJ
        self.ltd_decay = ltd_decay
        self.ltd_bins = int(window/defaultclock.dt)
        SpikeMonitor.__init__(self, IO)
        
    def propagate(self, spikes):
        '''
        Depress PF-PKJ synapses for GRs that spiked that synapse on
        PKJs that received CF spike.

        TO DO: consider dynamic time window so that the same
               GR spike doesn't get depressed twice due to 
               serial CF spikes and GR spike still in the same
               time window
        '''
        if len(spikes):
            # PKJs that received CF spike
            pkj_inds = LTD.postsynaptic_indexes(spikes,self.CF_PKJ)
            
            # GRs that synapse on pkj_inds
            gr_inds = LTD.presynaptic_indexes(pkj_inds,self.PF_PKJ)
            
            # GRs that spiked in the past ltd_bins
            gr_spiked_inds = self.GR.LS[:self.ltd_bins]

            # GRs that spiked and synapse on PKJs
            gr_ltd_inds = intersect1d(array(gr_inds),array(gr_spiked_inds))
            if not len(gr_ltd_inds):
                return
            
            # modify PF_PKJ synaptic strength
            self.PF_PKJ.w[gr_ltd_inds,pkj_inds] *= self.ltd_decay

    @staticmethod
    def presynaptic_indexes(neuron_inds,synapses):
        '''
        Map from target neuron indexes to source neuron indexes
        in synapses
        '''
        synapse_indexes = hstack(array(synapses.synapses_post)[neuron_inds])
        return synapses.presynaptic[synapse_indexes]
    
    @staticmethod
    def postsynaptic_indexes(neuron_inds,synapses):
        '''
        Map from source neuron indexes to target neuron indexes
        in synapses
        '''
        synapse_indexes = hstack(array(synapses.synapses_pre)[neuron_inds])
        return synapses.postsynaptic[synapse_indexes]