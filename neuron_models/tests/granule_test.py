'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = .25*ms

class GranuleTest(unittest.TestCase):

    def test_get_parameters(self):
        GR = GranuleCellGroup(3)
        self.assertEqual(GR.get_parameters()['N'],3)
        
    def test_model_equivalency(self):
        T = 200*msecond
        # spike train meant to cause neurons to spike
        spikes = rand(int(T/defaultclock.dt))
        spikes[spikes>.99] = 1.
        spikes[spikes<=.99] = 0.
        
        conn_weight_mfgr = 4.
        
        # Yamazaki implementation
        self.YGR = YamazakiNeuron(-35.,3.1,-58.,0.,-82.,-82.,.43,.18,0.028,
                                  1.,array([.88,.12]),array([.43,.57]),
                                  array([1.2,52]),array([7.,59.]),5.,0.,defaultclock.dt/ms)  
        
        # run Yamazaki implementation
        GR_spikes = []
        GR_V = [self.YGR.u]
        for s in spikes:
            GR_spikes.append(self.YGR.update(s,0,conn_weight_mfgr,0.))
            GR_V.append(self.YGR.u)
        
        # BRIAN Implementation
        GR = GranuleCellGroup(1)
        GR.V = GR.El
        GR.gahp = 0. * nsiemens
        
        # run BRIAN Implementation
        MF = SpikeGeneratorGroup(1,[(0,t*defaultclock.dt) for t in nonzero(spikes)[0]])
        S_GR_GR = Synapses(MF,GR,model='w:1',pre='g_ampa+=GR.g_ampa_*conn_weight_mfgr;g_nmda+=GR.g_ampa_*conn_weight_mfgr')
        S_GR_GR.connect_one_to_one()
        M_V = StateMonitor(GR,'V',record=0)
        
        run(T)
        
        #M_V.plot()
        #plot(M_V.times,array(GR_V[:-1])*mV)
        #show()
        
        self.assertAlmostEqual(norm(array(GR_V)[:-1] - M_V[0]/mV), 0., 10)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()