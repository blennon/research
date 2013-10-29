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
       
    def test_model_equivalency(self):
        T = 200*msecond
        # spike train meant to cause neurons to spike
        spikes = rand(int(T/defaultclock.dt))
        spikes[spikes>.95] = 1.
        spikes[spikes<=.95] = 0.
        
        # Yamazaki implementation
        self.YIO = YamazakiNeuron(Vth=-50., Cm=1., El=-60., Eex=0., Einh=-75., Eahp=-70., gl=.015, g_ex_=.1, g_inh_=.018,
                         g_ahp_=1., r_ex=array([1.0]), r_inh=array([1.0]), tau_ex=10., tau_inh=10., tau_ahp=5., I_spont=0.0,
                         dt=defaultclock.dt/ms) 
        
        conn_weight_gogr = 1.
        
        # run Yamazaki implementation
        IO_spikes = []
        IO_V = [self.YIO.u]
        for s in spikes:
            IO_spikes.append(self.YIO.update(s,0,conn_weight_gogr,0.,reset_V=True))
            IO_V.append(self.YIO.u)
        
        # BRIAN Implementation
        IO = InferiorOliveGroup(1)
        IO.V = IO.El
        IO.gahp = 0. * nsiemens
        
        # run BRIAN Implementation
        GR = SpikeGeneratorGroup(1,[(0,t*defaultclock.dt) for t in nonzero(spikes)[0]])
        S_GR_IO = Synapses(GR,IO,model='w:1',pre='g_ampa+=IO.g_ampa_*conn_weight_gogr')
        S_GR_IO.connect_one_to_one()
        M_V = StateMonitor(IO,'V',record=0)
        
        run(200*ms)
        
        M_V.plot()
        plot(M_V.times,array(IO_V[:-1])*mV,color='g')
        show()
        
        self.assertAlmostEqual(norm(array(IO_V)[:-1] - M_V[0]/mV), 0., 10)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()