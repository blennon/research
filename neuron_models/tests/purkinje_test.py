'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *
defaultclock.dt = 1.*ms

class PurkinjeTest(unittest.TestCase):

    def test_similar_membrane_potential(self):
        '''
        This test ensures the membrane potential trace for the BRIAN
        simulator implementation is very close to that implemented by
        Yamazaki and Nagao 2012
        '''
        w_ex = 1.
        
        T = 200*msecond
        # spike train meant to cause neurons to spike
        spikes = rand(int(T/defaultclock.dt))
        spikes[spikes>.9] = 1.
        spikes[spikes<=.9] = 0.      
        
        # Yamazaki implementation
        YPKJ = YamazakiNeuron(Vth=-55., Cm=106., El=-68., Eex=0., Einh=-75., Eahp=-70., gl=2.32, g_ex_=.7, g_inh_=1.0,
                         g_ahp_=100., r_ex=array([1.0]), r_inh=array([1.0]), tau_ex=8.3, tau_inh=10., tau_ahp=2.5, I_spont=0.0,
                         dt=defaultclock.dt/ms)

        # run Yamazaki
        YPKJ_spikes = []
        YPKJ_V = [YPKJ.u]
        for s in spikes:
            YPKJ_spikes.append(YPKJ.update(s,0,w_ex,0.,False))
            YPKJ_V.append(YPKJ.u)
            
        # BRIAN Implementation
        PKJ = PurkinjeCellGroup(1)
        PKJ.V = PKJ.El
        PKJ.g_ahp = 0. * nsiemens
        GR = SpikeGeneratorGroup(1,[(0,t*defaultclock.dt) for t in nonzero(spikes)[0]])
        S_GR_PKJ = Synapses(GR,PKJ,model='w:1',pre='g_ampa+=PKJ.g_ampa_*w_ex')
        S_GR_PKJ.connect_one_to_one()
        
        M_V = StateMonitor(PKJ,'V',record=0)
        M_spikes = SpikeMonitor(PKJ)
        run(200*ms)
        
        M_V.plot()
        plot(M_V.times,array(YPKJ_V[:-1])*mV)
        show()
        
        self.assertAlmostEqual(norm(M_V[0]/mV-array(YPKJ_V)[:-1]), 0., 10)

        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()