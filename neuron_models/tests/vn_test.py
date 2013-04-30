'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = 1.*ms

class VNTest(unittest.TestCase):
       
    def test_parameters(self):
        VN = VestibularNucleusGroup(2)
        params = VN.get_parameters()
        self.assertEquals(len(params),20)
        self.assertEquals(params['N'], 2)
        self.assertEquals(params['Vth'], -38.8 * mV)
        self.assertEquals(params['Cm'], 122.3 * pF)
        self.assertEquals(params['gl'], 1./.61 * nS)
        self.assertEquals(params['El'], -56. * mV)
        self.assertEquals(params['g_ampa_'], 50. * nS)
        self.assertEquals(params['g_nmda_'], 25.8 * nS)
        self.assertEquals(params['Eex'], 0. * mV)
        self.assertEquals(params['tau_ampa'], 9.9 * ms)
        self.assertEquals(params['tau_nmda'], 30.5 * ms)
        self.assertEquals(params['g_gaba_'], 30. * nS)
        self.assertEquals(params['Einh'], -88. * mV)
        self.assertEquals(params['tau_gaba'], 42.3 * ms)
        self.assertEquals(params['g_ahp_'], 50. * nS)
        self.assertEquals(params['Eahp'], -70. * mV)
        self.assertEquals(params['tau_ahp'], 5. * ms)
        self.assertEquals(params['r_ampa'], .66)
        self.assertEquals(params['r_nmda'], 1-.66)
        self.assertEquals(params['I_spont'], .7 * nA)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()