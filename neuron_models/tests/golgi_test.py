'''
Created on Apr 11, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = 1.*ms

class GolgiTest(unittest.TestCase):
    '''
    The purpose of this test is to make sure BRIAN's implementation
    of the neuron model is very close to Yamazaki's implementation
    '''

    def test_model_equivalency(self):
        # spike train meant to cause neurons to spike
        spikes = rand(200)
        spikes[spikes>.4] = 1.
        spikes[spikes<=.4] = 0.
        
        conn_weight_gogr = 2./(49.*100)
        
        # Yamazaki implementation
        self.YGO = YamazakiNeuron(-52.,28.,-55.,0.,0.,-72.7,2.3,45.5,0.0,
                                  20.,array([.8,.2*.33,.2*.67]),array([0.0]),
                                  array([1.5,31.,170.]),None,5.,0.,1.)  
        
        # run Yamazaki implementation
        GO_spikes = []
        GO_V = [self.YGO.u]
        for s in spikes:
            GO_spikes.append(self.YGO.update(s,0,conn_weight_gogr,0.))
            GO_V.append(self.YGO.u)
        
        # BRIAN Implementation
        GO = GolgiCellGroup(1)
        GO.V = GO.El
        GO.gahp = 0. * nsiemens
        
        # run BRIAN Implementation
        GR = SpikeGeneratorGroup(1,[(0,t * ms) for t in nonzero(spikes)[0]])
        S_GR_GO = Synapses(GR,GO,model='w:1',pre='g_ampa+=GO.g_ampa_*conn_weight_gogr;g_nmda1+=GO.g_ampa_*conn_weight_gogr;g_nmda2+=GO.g_ampa_*conn_weight_gogr')
        S_GR_GO.connect_one_to_one()
        M_V = StateMonitor(GO,'V',record=0)
        
        run(200*ms)
        
        self.assertAlmostEqual(norm(array(GO_V)[:-1] - M_V[0]/mV), 0., 10)
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()