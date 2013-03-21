'''
Created on Mar 21, 2013

@author: bill
'''
import unittest
from neuron_models import *


class ConnectionsTest(unittest.TestCase):

    def test_gr_go_connections(self):
        cxns = gr_to_go_connections(N_go = 32**2, N_gr = 32**2 * 10**2,dist=0,p=1.)
        self.assertEqual(len(cxns), 102400)
        self.assertEqual(cxns[10], (10,0))
        cxns = gr_to_go_connections(N_go = 5**2, N_gr = 5**2 * 10**2,dist=1,p=1.)
        for i in [0,1,2,5,6,7,10,11,12]:
            self.assertTrue((i*100,6) in cxns)
        for i in [3,8,13]:
            self.assertFalse((i*100,6) in cxns)
        cxns = gr_to_go_connections(N_go = 3**2, N_gr = 3**2 * 4**2,dist=1,p=1.)
        self.assertEqual(len(cxns), (4*4 + 6*4 + 9)*(4**2))
        
    def test_go_gr_connections(self):
        cxns = go_to_gr_connections(N_go=8**2,N_gr=8**2,dist=8,p=1.)
        self.assertEqual(len(cxns),8**4)
        cxns = go_to_gr_connections(N_go = 3**2, N_gr = 3**2 * 10, dist = 1, p=1.)
        for i in [0,1,3,4]: self.assertTrue((i,0) in cxns)
        self.assertFalse((2,0) in cxns)
        for i in range(9): self.assertTrue((i,80))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()