'''
Created on Mar 21, 2013

@author: bill
'''
import unittest
from neuron_models import *

class ConnectionsTest(unittest.TestCase):

    def test_gr_go_connections(self):
        pre, post = gr_to_go_connections(N_go = 32**2, N_gr = 32**2 * 10**2,dist=0,p=1.)
        cxns = zip(pre,post)
        self.assertEqual(len(cxns), 102400)
        self.assertTrue((10,0) in cxns)
        pre, post = gr_to_go_connections(N_go = 5**2, N_gr = 5**2 * 10**2,dist=1,p=1.)
        cxns = zip(pre,post)
        for i in [0,1,2,5,6,7,10,11,12]:
            self.assertTrue((i*100,6) in cxns)
        for i in [3,8,13]:
            self.assertFalse((i*100,6) in cxns)
        pre, post = gr_to_go_connections(N_go = 3**2, N_gr = 3**2 * 4**2,dist=1,p=1.)
        cxns = zip(pre,post)
        self.assertEqual(len(cxns), (4*4 + 6*4 + 9)*(4**2))
        
        # test connections with wrapping
        pre, post = gr_to_go_connections(N_go = 5**2, N_gr = 5**2 * 10**2,dist=1,p=1.,wrap=True)
        cxns = zip(pre,post)
        for i in [0,1,5,6,4,9,20,21,24]:
            self.assertTrue((i*100,0) in cxns)
        for i in [2,17,23]:
            self.assertFalse((i*100,0) in cxns)
        for i in [0,1,2,5,6,7,10,11,12]:
            self.assertTrue((i*100,6) in cxns)
        for i in [3,8,13]:
            self.assertFalse((i*100,6) in cxns)
        
    def test_go_gr_connections(self):
        pre, post = go_to_gr_connections(N_go=8**2,N_gr=8**2,dist=8,p=1.)
        cxns = zip(pre,post)
        self.assertEqual(len(cxns),8**4)
        pre, post = go_to_gr_connections(N_go = 3**2, N_gr = 3**2 * 10, dist = 1, p=1.)
        cxns = zip(pre,post)
        for i in [0,1,3,4]: self.assertTrue((i,0) in cxns)
        self.assertFalse((2,0) in cxns)
        for i in range(9): self.assertTrue((i,80))
        
        # test connections with wrapping
        pre, post = go_to_gr_connections(N_go = 5**2, N_gr = 5**2 * 10, dist = 1, p=1., wrap=True)
        cxns = zip(pre,post)
        for i in [0,3,4,15,20,18,19,23,24]: self.assertTrue((i,240) in cxns)
        self.assertFalse((5,240) in cxns)

    def test_gr_pkj_connections(self):
        pre, post = gr_to_pkj_connections(5**2,2**2 * 5**2,3,1)
        cxns = zip(pre,post)
        self.assertEqual(len(cxns),3*3*20)
        for i in [0,80,20]:
            self.assertTrue((i,0) in cxns)
        for i in [20,40]:
            self.assertFalse((i,2) in cxns)
    
    def test_gr_mli_connections(self):
        N_GO = 5**2
        N_GR = N_GO * 2**2
        N_PKJ = 3
        MLI_PKJ_ratio = 2
        pre, post = gr_to_mli_connections(N_GO,N_GR,N_PKJ,MLI_PKJ_ratio,go_span=1,p=1)
        cxns = zip(pre,post)
        self.assertEqual(len(cxns),3*3*20*2)
        for c in [(0,0),(0,1),(80,0),(80,1),(20,0),(20,1)]:
            self.assertTrue(c in cxns)
        for i in [20,40]:
            self.assertFalse((i,5) in cxns)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()