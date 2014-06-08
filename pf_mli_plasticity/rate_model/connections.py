__author__ = 'bill'
from pylab import *

class Connection(object):
    '''
    Defines a connection object between two neuron groups

    the 'state' is the connection matrix
    '''

    def __init__(self, src, trg, W=None, type='excitatory'):
        '''
        src: source neuron group
        trg: target neuron group
        W: connection matrix, set to zeros if None. dimensions should be (num trg neurons, num src neurons)
        '''
        self.src, self.trg = src, trg
        self.state = zeros((trg.get_num_neurons(),src.get_num_neurons()))
        if W is not None:
            if W.shape != self.state.shape:
                raise Exception('Dimensions of W do not match (%s,%s)' % self.state.shape)
            self.state = W
        self.type = type

    def update(self):
        '''
        update the connection matrix
        '''
        raise NotImplementedError

    def get_state(self):
        '''
        returns the connection matrix
        '''
        return self.state

    def get_source(self):
        '''
        return the source neuron group
        '''
        return self.src

    def get_source_state(self):
        '''
        return the state of the source neuron group
        '''
        return self.src.get_state()

    def get_type(self):
        return self.type

    def get_synapse_polarity(self):
        '''
        returns 1 if excitatory synapse, -1 otherwise
        '''
        if self.type == 'excitatory':
            return 1.
        return -1.

class PF_MLI_Connection(Connection):
    '''
    Defines the connection between PFs and MLIs
    '''
    def __init__(self, src, trg, W=None):
        super(PF_MLI_Connection,self).__init__(src, trg, W)

    def update(self, beta=.01):
        '''
        define the update rule for this connection

        dW/dt = B(MLI - w)PF if w>0
        '''
        MLI = self.trg.get_state()[...,None]
        PF = self.src.get_state()[...,None]
        dw_dt = beta*(MLI - self.state)*PF.T
        # only update active synapses, i.e. w > 0
        self.state[self.state>0] += dw_dt[self.state>0]