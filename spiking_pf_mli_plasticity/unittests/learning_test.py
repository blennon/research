'''
Created on Mar 21, 2013

@author: bill
'''
import unittest
from neuron_models import *
from spiking_pf_mli_plasticity import *
from brian import *

class LearningTest(unittest.TestCase):

    def test_compute_alpha(self):
        MLI = MLIGroup(4)
        CF = MLIGroup(2)
        S_CF_MLI = Synapses(CF,MLI,model='''w:1''')
        S_CF_MLI[0,0] = True
        S_CF_MLI[0,1] = True
        S_CF_MLI[1,1] = True
        S_CF_MLI[1,2] = True
        S_CF_MLI.w[:,:] = 1.
        run(1*msecond)

        class CF_R:
            def get_normalized_firing_rates(self,CF_max):
                return array([.2,.5])

        alpha = compute_alpha(CF_R(), S_CF_MLI)
        self.assertEquals(norm(array([ 0.9 ,  0.65,  0.75,  1.  ]) - alpha), 0)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()