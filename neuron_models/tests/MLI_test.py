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

    def test_model_equivalency(self):
        T = 200*msecond
        # spike train meant to cause neurons to spike
        spikes = rand(int(T/defaultclock.dt))
        spikes[spikes>.95] = 1.
        spikes[spikes<=.95] = 0.
        
        # Yamazaki implementation
        self.YMLI = YamazakiNeuron(-53.,14.6,-68.,0.,-82.,-82.,1.6,1.3,4.,
                                  50.,array([1.]),array([1.0]),
                                  array([.8]),array([4.6]),2.5,0.,defaultclock.dt/ms)  
        
        conn_weight_gogr = 1.
        
        # run Yamazaki implementation
        MLI_spikes = []
        MLI_V = [self.YMLI.u]
        for s in spikes:
            MLI_spikes.append(self.YMLI.update(s,0,conn_weight_gogr,0.,reset_V=False))
            MLI_V.append(self.YMLI.u)
        
        # BRIAN Implementation
        MLI = MLIGroup(1)
        MLI.V = MLI.El
        MLI.gahp = 0. * nsiemens
        
        # run BRIAN Implementation
        GR = SpikeGeneratorGroup(1,[(0,t*defaultclock.dt) for t in nonzero(spikes)[0]])
        S_GR_MLI = Synapses(GR,MLI,model='w:1',pre='g_ampa+=MLI.g_ampa_*conn_weight_gogr')
        S_GR_MLI.connect_one_to_one()
        M_V = StateMonitor(MLI,'V',record=0)
        
        run(200*ms)
        
        M_V.plot()
        plot(M_V.times,array(MLI_V[:-1])*mV,color='g')
        show()
        
        self.assertAlmostEqual(norm(array(MLI_V)[:-1] - M_V[0]/mV), 0., 10)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()