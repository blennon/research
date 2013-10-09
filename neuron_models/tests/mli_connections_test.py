'''
Created on Mar 21, 2013

@author: bill
'''
import unittest
from neuron_models import *

class MLIConnectionsTest(unittest.TestCase):

    def test_mli_mli_dir(self):
        MLI = MLIGroup(4)
        S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w_mli_mli')
               
        # all connections should go to the right
        connect_mli_mli(S_MLI_MLI,2,1,0)
        self.assertTrue(len(S_MLI_MLI.synapse_index((3,0))))
        self.assertTrue(len(S_MLI_MLI.synapse_index((3,1))))
        self.assertFalse(len(S_MLI_MLI.synapse_index((3,2))))
        
        S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w_mli_mli')
               
        # all connections should go to the left
        connect_mli_mli(S_MLI_MLI,2,1,1)
        self.assertFalse(len(S_MLI_MLI.synapse_index((3,0))))
        self.assertTrue(len(S_MLI_MLI.synapse_index((3,1))))
        self.assertTrue(len(S_MLI_MLI.synapse_index((3,2))))
        
    def test_mli_mli_num(self):
        MLI = MLIGroup(10)
        S_MLI_MLI = Synapses(MLI,MLI,model='w:1',pre='g_inh+=MLI.g_inh_*w_mli_mli')
        connect_mli_mli(S_MLI_MLI,3,1,.5)
        
        # should be three synapses
        self.assertEquals(len(S_MLI_MLI),30)
        
    def test_mli_pkj(self):
        MLI = MLIGroup(6)
        PKJ = MLIGroup(3)
        S_MLI_PKJ = Synapses(MLI,PKJ,model='w:1',pre='g_inh+=MLI.g_inh_*w_mli_mli')
        connect_mli_pkj(S_MLI_PKJ, 2, 1, 0)
        self.assertEquals(len(S_MLI_PKJ), 12)
        for i in range(3):
            self.assertTrue(len(S_MLI_PKJ.synapse_index((2*i,i))))
            self.assertTrue(len(S_MLI_PKJ.synapse_index((2*i+1,i))))
            self.assertTrue(len(S_MLI_PKJ.synapse_index((2*i,(i+1)%3))))
            self.assertTrue(len(S_MLI_PKJ.synapse_index((2*i+1,(i+1)%3))))
            self.assertFalse(len(S_MLI_PKJ.synapse_index((2*i,(i+2)%3))))
            self.assertFalse(len(S_MLI_PKJ.synapse_index((2*i+1,(i+2)%3))))
            
    def test_mli_pkj_rand(self):
        MLI = MLIGroup(300)
        PKJ = MLIGroup(30)
        S_MLI_PKJ = Synapses(MLI,PKJ,model='w:1',pre='g_inh+=MLI.g_inh_*w_mli_mli')
        connect_mli_pkj(S_MLI_PKJ, 10, .5, .5)
        self.assertTrue(len(S_MLI_PKJ)<2000)  
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()