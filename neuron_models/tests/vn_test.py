'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = .5*ms

class VNTest(unittest.TestCase):
       
    def test_model_equivalency(self):
        T = 200*msecond
        # spike train meant to cause neurons to spike
        spikes = rand(int(T/defaultclock.dt))
        spikes[spikes>.95] = 1.
        spikes[spikes<=.95] = 0.
        
        # Yamazaki implementation
        self.YVN = YamazakiNeuron(Vth=-38.8, Cm=122.3, El=-56., Eex=0., Einh=-88., Eahp=-70., gl=1./.61, g_ex_=50., g_inh_=30.,
                         g_ahp_=50., r_ex=array([.66,.34]), r_inh=array([1.0]), tau_ex=array([9.9,30.5]), tau_inh=42.3, tau_ahp=5., I_spont=700.,
                         dt=defaultclock.dt/ms) 
        
        conn_weight = 0.05
        
        # run Yamazaki implementation
        VN_spikes = []
        VN_V = [self.YVN.u]
        for s in spikes:
            VN_spikes.append(self.YVN.update(s,0,conn_weight,0.,reset_V=False))
            VN_V.append(self.YVN.u)
        
        # BRIAN Implementation
        VN = VestibularNucleusGroup(1)
        VN.V = VN.El
        VN.gahp = 0. * nsiemens
        
        # run BRIAN Implementation
        GR = SpikeGeneratorGroup(1,[(0,t*defaultclock.dt) for t in nonzero(spikes)[0]])
        S_GR_VN = Synapses(GR,VN,model='w:1',pre='g_ampa+=VN.g_ampa_*conn_weight; g_nmda+=VN.g_ampa_*conn_weight')
        S_GR_VN.connect_one_to_one()
        M_V = StateMonitor(VN,'V',record=0)
        
        run(200*ms)
        
        #M_V.plot()
        #plot(M_V.times,array(VN_V[:-1])*mV,color='g')
        #show()
        
        self.assertAlmostEqual(norm(array(VN_V)[:-1] - M_V[0]/mV), 0., 10)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()