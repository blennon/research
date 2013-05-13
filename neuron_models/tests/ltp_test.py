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

class LTPTest(unittest.TestCase):
    '''
    This tests CF driven LTP
    '''

    def setUp(self):
        T = 200*ms
        self.ltp_inc = .1
        self.max_weight = 1.
        ltp_window = 50*ms
        GR_spiketimes = [(0,50*ms),(0,98*ms),
                         (1,30*ms),(1,32*ms),(1,130*ms)]
        IO_spiketimes = [(0,100*ms),(0,160*ms),
                         (1,49*ms),(1,75*ms)]
        GR = SpikeGeneratorGroupDelay(2,GR_spiketimes,max_delay=ltp_window)
        IO = SpikeGeneratorGroup(2,IO_spiketimes)
        BS = BasketCellGroup(2)
        BS.g_ahp = 0
        BS.V = BS.El

        CF_BS = Synapses(IO,BS,model='''w:1''',pre='''g_ampa+=BS.g_ampa_*w''')
        PF_BS = Synapses(GR,BS,model='''w:1''',pre='''g_ampa+=BS.g_ampa_*w''')
        CF_BS[0,0] = 1.
        CF_BS[1,1] = 1.
        PF_BS[0,0] = 1.
        PF_BS[1,1] = 1.
        PF_BS[0,1] = 1.
        CF_BS.w = 0.5
        PF_BS.w = 0.1
        LTP = PF_MLI_LTP(IO, GR, PF_BS, CF_BS, self.ltp_inc, self.max_weight, window=ltp_window)
        
        self.M_V = StateMonitor(BS,'V',record=[0,1])
        self.M_IO_spikes = SpikeMonitor(IO)
        self.M_GR_spikes = SpikeMonitor(GR)
        self.M_w = StateMonitor(PF_BS,'w',record=[0,1,2])
        run(T)
        
    def test_ltd(self):
        self.assertEquals(self.M_w[0][0],.1)
        self.assertEquals(self.M_w[0][200],.1 + (self.max_weight-.1)*self.ltp_inc)
        self.assertEquals(self.M_w[1][0],.1)
        self.assertEquals(self.M_w[1][100],.1 + (self.max_weight-.1)*self.ltp_inc*2)
        self.assertEquals(self.M_w[2][0],.1)
        self.assertEquals(self.M_w[2][150],.1 + (self.max_weight-.1)*self.ltp_inc)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()