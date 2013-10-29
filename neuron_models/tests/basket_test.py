'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = 1.*ms

class BasketTest(unittest.TestCase):

    def test_similar_membrane_potential(self):
        '''
        This test ensures the membrane potential trace for the BRIAN
        simulator implementation is very close to that implemented by
        Yamazaki and Nagao 2012
        '''
        w_ex = .3
        
        # stimulus
        spikes = rand(200)
        spikes[spikes>.6] = 1.
        spikes[spikes<=.6] = 0.        
        
        # Yamazaki implementation
        YBS = YamazakiNeuron(Vth=-55., Cm=106., El=-68., Eex=0., Einh=0., Eahp=-70., gl=2.32, g_ex_=.7, g_inh_=0.,
                         g_ahp_=100., r_ex=array([1.0]), r_inh=array([0.0]), tau_ex=8.3, tau_inh=None, tau_ahp=2.5, I_spont=0.,
                         dt=1.)

        # run Yamazaki
        YBS_spikes = []
        YBS_V = [YBS.u]
        for s in spikes:
            YBS_spikes.append(YBS.update(s,0,w_ex,0.,False))
            YBS_V.append(YBS.u)
            
        # BRIAN Implementation
        BS = BasketCellGroup(1)
        BS.V = BS.El
        BS.g_ahp = 0. * nsiemens
        GR = SpikeGeneratorGroup(1,[(0,t * ms) for t in nonzero(spikes)[0]])
        S_GR_BS = Synapses(GR,BS,model='w:1',pre='g_ampa+=BS.g_ampa_*w_ex')
        S_GR_BS.connect_one_to_one()
        
        M_V = StateMonitor(BS,'V',record=0)
        M_spikes = SpikeMonitor(BS)
        run(200*ms)
        
        self.assertAlmostEqual(norm(M_V[0]/mV-array(YBS_V)[:-1]), 0., 10)
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()