from pylab import *
from brian import *

class CFDrivenLearning(SpikeMonitor):
    
    def propogate(self):
        raise NotImplementedError
    
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

    
class PF_PKJ_LTD(CFDrivenLearning):
    '''
    Implements LTD on PF-PKJ synapses driven by CF activity
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
            pkj_inds = PF_PKJ_LTD.postsynaptic_indexes(spikes,self.CF_PKJ)
            
            # GRs that synapse on pkj_inds
            gr_inds = PF_PKJ_LTD.presynaptic_indexes(pkj_inds,self.PF_PKJ)
            
            # GRs that spiked in the past ltd_bins
            gr_spiked_inds = self.GR.LS[:self.ltd_bins]
            # number of times each gr spiked
            gr_spike_counts = bincount(gr_spiked_inds)
            
            # GRs that spiked and synapse on PKJs
            gr_ltd_inds = intersect1d(array(gr_inds),array(gr_spiked_inds))
            if not len(gr_ltd_inds):
                return
            
            # counts of gr spikes for grs undergoing ltd
            gr_ltd_spike_counts = gr_spike_counts[gr_ltd_inds]
            
            # depress PF-PKJ synapse proportionally to the number of
            # times the PF fired in the past self.ltd_bins
            for i in unique(gr_ltd_spike_counts):
                if i == 0: continue
                self.PF_PKJ.w[gr_ltd_inds[gr_ltd_spike_counts==i],pkj_inds] *= self.ltd_decay**i

class PF_MLI_LTP(CFDrivenLearning):
    '''
    Implements additive LTP on PF-MLI synapses driven by CF activity
    '''
    def __init__(self, IO, GR, PF_MLI, CF_MLI, ltp_inc, max_weight, window=50*ms):
        '''
        IO: NeuronGroup of inferior olive neurons
        GR: NeuronGroup of granule cells
        PF_MLI: Synapses object of PF-MLI synapses
        CF_MLI: Synapses object of CF-MLI synapses
        ltp_inc: the additive constant to increment PF-MLI synapses by
        window: the time window to depress PF-PKJ synapses for GR spikes 
                that precede IO spikes.
        '''
        self.IO = IO
        self.GR = GR
        self.PF_MLI = PF_MLI
        self.CF_MLI = CF_MLI
        self.ltp_inc = ltp_inc
        self.max_weight = max_weight
        self.ltp_bins = int(window/defaultclock.dt)
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
            # MLIs that received CF spike
            mli_inds = PF_MLI_LTP.postsynaptic_indexes(spikes,self.CF_MLI)
            
            # GRs that synapse on mli_inds
            gr_inds = PF_MLI_LTP.presynaptic_indexes(mli_inds,self.PF_MLI)
            
            # GRs that spiked in the past ltd_bins
            gr_spiked_inds = self.GR.LS[:self.ltp_bins]
            # number of times each gr spiked
            gr_spike_counts = bincount(gr_spiked_inds)
            
            # GRs that spiked and synapse on MLIs
            gr_ltp_inds = intersect1d(array(gr_inds),array(gr_spiked_inds))
            if not len(gr_ltp_inds):
                return
            
            # counts of gr spikes for grs undergoing ltp
            gr_ltp_spike_counts = gr_spike_counts[gr_ltp_inds]
            
            # increase PF-MLI synapse proportionally to the number of
            # times the PF fired in the past self.ltp_bins
            for i in unique(gr_ltp_spike_counts):
                if i == 0: continue
                curr_weights = self.PF_MLI.w[gr_ltp_inds[gr_ltp_spike_counts==i],mli_inds]
                self.PF_MLI.w[gr_ltp_inds[gr_ltp_spike_counts==i],mli_inds] += (self.max_weight-curr_weights)*self.ltp_inc*i