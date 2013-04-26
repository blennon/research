'''
Created on Apr 25, 2013

@author: bill
'''
import unittest

import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = .5*ms

class LTDTest(unittest.TestCase):


    def setUp(self):
        T = 200*ms
        ltd_window = 50*ms
        GR_spiketimes = [(0,50*ms),(0,98*ms),
                         (1,30*ms),(1,70*ms),(1,130*ms)]
        IO_spiketimes = [(0,100*ms),(0,160*ms),
                         (1,49*ms),(1,75*ms)]
        GR = SpikeGeneratorGroupDelay(2,GR_spiketimes,max_delay=ltd_window)
        IO = SpikeGeneratorGroup(2,IO_spiketimes)
        PKJ = PurkinjeCellGroup(2)
        PKJ.g_ahp = 0
        PKJ.V = PKJ.El

        CF_PKJ = Synapses(IO,PKJ,model='''w:1''',pre='''g_ampa+=PKJ.g_ampa_*w''')
        PF_PKJ = Synapses(GR,PKJ,model='''w:1''',pre='''g_ampa+=PKJ.g_ampa_*w''')
        CF_PKJ[0,0] = 1.
        CF_PKJ[1,1] = 1.
        PF_PKJ[0,0] = 1.
        PF_PKJ[1,1] = 1.
        PF_PKJ[0,1] = 1.
        CF_PKJ.w = 0.5
        PF_PKJ.w = 0.1
        PF_PKJ_LTD = LTD(IO,GR,PF_PKJ,CF_PKJ,.995,ltd_window)
        
        self.M_V = StateMonitor(PKJ,'V',record=[0,1])
        self.M_IO_spikes = SpikeMonitor(IO)
        self.M_GR_spikes = SpikeMonitor(GR)
        self.M_w = StateMonitor(PF_PKJ,'w',record=[0,1,2])
        run(T)
        
    def test_ltd(self):
        self.assertEquals(self.M_w[0][0],.1)
        self.assertEquals(self.M_w[0][200],.1*.995)
        self.assertEquals(self.M_w[1][0],.1)
        self.assertEquals(self.M_w[1][100],.1*.995)
        self.assertEquals(self.M_w[1][150],.1*.995*.995)
        self.assertEquals(self.M_w[2][0],.1)
        self.assertEquals(self.M_w[2][150],.1*.995)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()