__author__ = 'bill'

import unittest
from rate_model.neuron_group import *
from rate_model.connections import *

class TestNeuronGroup(unittest.TestCase):

    def test_neuron_group_up(self):
        PF = NeuronGroup(3, resting_state=array([1,1,0]))
        self.assertEquals(PF.get_num_neurons(),3)
        self.assertEquals(PF.get_state()[0], 1)

    def test_update(self):
        MLI = NeuronGroup(1, array([1]))
        PF = NeuronGroup(3, array([1,1,1]))
        C_PF_MLI = Connection(PF,MLI, array([[1,1,1]]))
        MLI.connect(C_PF_MLI)
        MLI.update()
        self.assertEquals(MLI.get_state()[0], 4)


if __name__ == '__main__':
    unittest.main()
