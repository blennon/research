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

    def setUp(self):
        # Neuron parameters
        C_GO = 28.0
        R_GO = 0.428
        E_leak_GO = -55.0
        g_leak_GO = 2.3#(1.0/(R_GO))
        E_ex_GO = 0.0
        g_ex_GO = 45.5
        E_ahp_GO = -72.7
        g_ahp_GO = 20.0
        tau_ampa_gogr = 1.5
        tau1_nmda_gogr = 31.0
        tau2_nmda_gogr = 170.0
        r_ampa_gogr = 0.8
        r1_nmda_gogr = (0.2*0.33)
        r2_nmda_gogr = (0.2*0.67)
        tau_ahp_go = 5.0
        TH_GO = -52.
        
        dt = 1.
        decay_ampa_gogr = exp(-dt/tau_ampa_gogr)
        decay1_nmda_gogr = exp(-dt/tau1_nmda_gogr)
        decay2_nmda_gogr = exp(-dt/tau2_nmda_gogr)
        decay_ahp_go = exp(-dt/tau_ahp_go)
        
        conn_weight_gogr = 2./(49.*100)

        def update_u(u,g_gogr,ahp_go):
            '''membrane update equation'''
            dudt = (1./C_GO)*(-g_leak_GO*(u-E_leak_GO)
                            -g_ex_GO*g_gogr*(u-E_ex_GO)
                            -g_ahp_GO*ahp_go*(u-E_ahp_GO))
            return u + dudt*dt
            
        # create some fictitious spike input
        spikes = zeros(200)
        spikes[50] = 1.
        spikes[100] = 1.
        spikes[150]=1.
        
        # Simulate Yamazaki implementation
        u = [E_leak_GO]
        psp_gr_ampa = [0.]
        psp_gr_nmda1 = [0.]
        psp_gr_nmda2 = [0.]
        psp_gr = [0.]
        ahp_go = [0.]
        g_gogr = [0.]
        GO_spikes = [0.]
        for s in spikes:
            psp_gr_ampa.append(psp_gr_ampa[-1] * decay_ampa_gogr + s)
            psp_gr_nmda1.append(psp_gr_nmda1[-1] * decay1_nmda_gogr + s)
            psp_gr_nmda2.append(psp_gr_nmda2[-1] * decay2_nmda_gogr + s)
            psp_gr.append(r_ampa_gogr*psp_gr_ampa[-1]+
                          r1_nmda_gogr*psp_gr_nmda1[-1]+
                          r2_nmda_gogr*psp_gr_nmda2[-1])
            g_gogr.append(conn_weight_gogr*psp_gr[-1])
            
            if GO_spikes[-1] == 1.:
                ahp_go.append(1.)
            else:
                ahp_go.append(ahp_go[-1]*decay_ahp_go)
                
            new_u = update_u(u[-1],g_gogr[-1],ahp_go[-1])
            if new_u > TH_GO:
                u.append(E_leak_GO)
                GO_spikes.append(1.)
            else:
                u.append(new_u)
                GO_spikes.append(0.)
        self. u1 = array(u[:-1])
        
        
        # BRIAN Implementation
        GO = GolgiCellGroup(1)
        GO.V = GO.El
        GO.gahp = 0. * nsiemens
        GR = SpikeGeneratorGroup(1,[(0,t * ms) for t in nonzero(spikes)[0]])
        S_GR_GO = Synapses(GR,GO,model='w:1',pre='g_ampa+=GO.g_ampa_*conn_weight_gogr;g_nmda1+=GO.g_ampa_*conn_weight_gogr;g_nmda2+=GO.g_ampa_*conn_weight_gogr')
        S_GR_GO.connect_one_to_one()
        M_V = StateMonitor(GO,'V',record=0)
        run(200*ms)
        self.u2 = M_V[0] / mV
        
    def tearDown(self):
        pass


    def testMembranePotentialsSimilarity(self):
        self.assertAlmostEqual(norm(self.u1 - self.u2), 0., 2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()