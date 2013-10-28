'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = .25*ms

class MLITest(unittest.TestCase):      

    def test_parameters(self):
        MLI = MLIGroup(2)
        params = MLI.get_parameters()
        self.assertEquals(params['N'], 2)
        self.assertEquals(params['Vth'], -53. * mV)
        self.assertEquals(params['Cm'], 14.6 * pF)
        self.assertEquals(params['gl'], 1.6 * nS)
        self.assertEquals(params['El'], -68. * mV)
        self.assertEquals(params['g_ampa_'], 1.3 * nS)
        self.assertEquals(params['Eex'], 0. * mV)
        self.assertEquals(params['tau_ampa'], .8 * ms)
        self.assertEquals(params['g_ahp_'], 50. * nS)
        self.assertEquals(params['Eahp'], -82. * mV)
        self.assertEquals(params['tau_ahp'], 2.5 * ms)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()