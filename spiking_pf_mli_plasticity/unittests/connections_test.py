'''
Created on Mar 21, 2013

@author: bill
'''
import unittest
from neuron_models import *
from spiking_pf_mli_plasticity import *
from brian import *

class ConnectionsTest(unittest.TestCase):

    def test_gr_mli_connections(self):
        N_MLI_groups=16
        N_MLI_per_group=10
        GR_cluster_width=3
        N_GR_clusters=15
        N_MLI=N_MLI_groups*N_MLI_per_group
        N_GR = N_GR_clusters*GR_cluster_width*N_MLI_groups
        GR = PoissonGroup(N_GR,rates=None)
        MLI = PoissonGroup(N_MLI,rates=None)
        S_GR_MLI = Synapses(GR,MLI,model='''w:1''')
        S_GR_MLI = connect_gr_mli(S_GR_MLI,N_MLI_groups,N_MLI_per_group, GR_cluster_width, N_GR_clusters)

        self.assertEquals((S_GR_MLI.w[:,:]>0).sum(), ((9*14 + 6*2)*10))
        self.assertEquals((S_GR_MLI.w[:,:]==0).sum(),(((45*3*14 + 45*2*2)*10) - ((9*14 + 6*2)*10)))
        active_syns=[(21,0),(22,0),(23,0),(22+45,10),(23+90,10)]
        inactive_syns=[(24,0),(45,10)]
        no_syn=[(0,20),(45,60)]
        for i,j in active_syns:
            self.assertGreater(S_GR_MLI.w[i,j],0)
        for i,j in inactive_syns:
            self.assertEquals(S_GR_MLI.w[i,j],0)
        for i,j in no_syn:
            self.assertEquals(S_GR_MLI.w[i,j].shape[0], 0)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()