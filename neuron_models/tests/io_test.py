'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = 1.*ms

class IOTest(unittest.TestCase):
       
    def test_parameters(self):
        IO = InferiorOliveGroup(2)
        params = IO.get_parameters()
        self.assertEquals(len(params),15)
        self.assertEquals(params['N'], 2)
        self.assertEquals(params['Vth'], -50. * mV)
        self.assertEquals(params['Cm'], 1. * pF)
        self.assertEquals(params['gl'], 0.015 * nS)
        self.assertEquals(params['El'], -60. * mV)
        self.assertEquals(params['g_ampa_'], .1 * nS)
        self.assertEquals(params['Eex'], 0. * mV)
        self.assertEquals(params['tau_ampa'], 10. * ms)
        self.assertEquals(params['g_gaba_'], .018 * nS)
        self.assertEquals(params['Einh'], -75. * mV)
        self.assertEquals(params['tau_gaba'], 10. * ms)
        self.assertEquals(params['g_ahp_'], 1. * nS)
        self.assertEquals(params['Eahp'], -70. * mV)
        self.assertEquals(params['tau_ahp'], 5. * ms)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()