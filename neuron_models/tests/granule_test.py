'''
Created on Apr 19, 2013

@author: bill
'''
import unittest
from pylab import *
from brian import *
from neuron_models import *


class GranuleTest(unittest.TestCase):

    def test_get_parameters(self):
        GR = GranuleCellGroup(3)
        self.assertEqual(GR.get_parameters()['N'],3)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()