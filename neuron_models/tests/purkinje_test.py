'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *

class PurkinjeTest(unittest.TestCase):

    def test_parameters(self):
        PKJ = PurkinjeCellGroup(2)
        params = PKJ.get_parameters()
        self.assertEquals(params['N'], 2)
        self.assertEquals(params['Vth'], -55. * mV)
        self.assertEquals(params['Cm'], 107. * pF)
        self.assertEquals(params['gl'], 2.32 * nS)
        self.assertEquals(params['El'], -68. * mV)
        self.assertEquals(params['g_ampa_'], .7 * nS)
        self.assertEquals(params['Eex'], 0. * mV)
        self.assertEquals(params['tau_ampa'], 8.3 * ms)
        self.assertEquals(params['g_inh_'], 1. * nS)
        self.assertEquals(params['Einh'], -75. * mV)
        self.assertEquals(params['tau_inh'], 10. * ms)
        self.assertEquals(params['g_ahp_'], .1 * nS)
        self.assertEquals(params['Eahp'], -70. * mV)
        self.assertEquals(params['tau_ahp'], 5. * ms)
        self.assertEquals(params['I_spont'], 0.25 * nA)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()