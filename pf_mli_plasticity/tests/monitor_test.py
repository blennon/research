__author__ = 'bill'

import unittest
from rate_model.neuron_group import *
from rate_model.connections import *
from rate_model.utilities import StateMonitor

class TestStateMonitor(unittest.TestCase):

    def test_update(self):
        MLI = NeuronGroup(1, array([1]))
        MLI_monitor = StateMonitor(MLI)
        PF = NeuronGroup(3, array([1,1,1]))
        C_PF_MLI = Connection(PF,MLI, array([[1,1,1]]))
        MLI.connect(C_PF_MLI)
        MLI.update()
        MLI_monitor.record()
        self.assertEquals(MLI_monitor.get_recording().sum(), 5)


if __name__ == '__main__':
    unittest.main()
