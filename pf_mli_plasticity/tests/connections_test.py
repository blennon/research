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
        C_PF_MLI = Connection(PF,MLI)
        self.assertEquals(C_PF_MLI.get_state().shape,(2,3))

    def test_build_pf_mli_connections_matrix(self):
        '''
        tests building the connection matrix for PF-MLI connections
        '''
        N_PF, N_MLI = 10, 2
        test_W = array([[ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.]])
        self.assertEquals(norm(build_pf_mli_connection_matrix(10,2) - test_W),0)

    def test_build_MLI_MLI_weight_matrix(self):
        N_MLI = 3
        test_W = array([[ 0.,  1.,  1.],
                        [ 1.,  0.,  1.],
                        [ 1.,  1.,  0.]])
        self.assertEquals(norm(build_MLI_MLI_weight_matrix(3,1.)-test_W),0)
"""
    # NEED TO UPDATE THIS SINCE MAKING CHANGES TO PF_MLI_Connection
    def test_update(self):
        '''
        tests the learning rule
        '''
        MLI = NeuronGroup(2, resting_state=array([1,0]))
        PF = NeuronGroup(3, resting_state=array([1,1,0]))
        W = array([[ 0.5,  0.5,  0.  ],[ 0.  ,  0.  ,  0.  ]])
        C_PF_MLI = PF_MLI_Connection(PF,MLI,W=W,PF_trace=PF)
        C_PF_MLI.update()
        W_ = array([[ 0.505,  0.505,  0.  ],[ 0.  ,  0.  ,  0.  ]])
        self.assertEquals((C_PF_MLI.get_state() - W_).sum(),0)
        self.assertEquals(C_PF_MLI.get_state()[0,0], W_[0,0])
"""


if __name__ == '__main__':
    unittest.main()
