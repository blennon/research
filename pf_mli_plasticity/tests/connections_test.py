__author__ = 'bill'

import unittest
from rate_model.connections import *
from rate_model.neuron_group import *

class TestConnections(unittest.TestCase):

    def test_connection_matrix_shape(self):
        '''
        tests the correct shape of the connection matrix
        '''
        MLI = NeuronGroup(2, resting_state=array([1,0]))
        PF = NeuronGroup(3, resting_state=array([1,1,0]))
        C_PF_MLI = PF_MLI_Connection(PF,MLI)
        self.assertEquals(C_PF_MLI.get_state().shape,(2,3))

    def test_update(self):
        '''
        tests the learning rule
        '''
        MLI = NeuronGroup(2, resting_state=array([1,0]))
        PF = NeuronGroup(3, resting_state=array([1,1,0]))
        W = array([[ 0.5,  0.5,  0.  ],[ 0.  ,  0.  ,  0.  ]])
        C_PF_MLI = PF_MLI_Connection(PF,MLI,W)
        C_PF_MLI.update()
        W_ = array([[ 0.505,  0.505,  0.  ],[ 0.  ,  0.  ,  0.  ]])
        self.assertEquals((C_PF_MLI.get_state() - W_).sum(),0)
        self.assertEquals(C_PF_MLI.get_state()[0,0], W_[0,0])

if __name__ == '__main__':
    unittest.main()
