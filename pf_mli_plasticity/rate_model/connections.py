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

    def get_target(self):
        '''
        return the target neuron group
        '''
        return self.trg

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
    def __init__(self, src, trg, PF_trace, W=None, delta=.1, dt=.1):
        '''
        src: source neuron group
        trg: target neuron group
        PF_trace: a neuron group recording a trace of the PF activity
        W: connection matrix
        delta: initial weight for inactive synapses
        '''
        super(PF_MLI_Connection,self).__init__(src, trg, W)
        self.delta = delta
        self.dt = dt
        self.PF_trace = PF_trace

    def update(self, beta=.001, CF_active=False):
        '''
        define the update rule for this connection

        dW/dt = B(MLI - w)PF, if w>0
        dW/dt = delta, if w=0 and CF>0, PF>0
        dW/dt = 0, otherwise

        CF_active: boolean if an impinging CF is active or not
        use_PF_trace: boolean, if True, use a moving average of the PF activity
        '''

        # set co-active PF-CF synapses to delta
        if CF_active:
            PF = self.src.get_state()
            self.state[(self.state == 0) & tile(PF>0,(self.state.shape[0],1))] = self.delta

        # update active synapses according to GSD
        PF, MLI = self.PF_trace.get_state()[...,None], self.trg.get_state()[...,None]
        dW_dt = beta*(MLI - 1.1*self.state)*PF.T
        # only update active synapses, i.e. w > 0
        self.state[self.state>0] += self.dt*dW_dt[self.state>0]