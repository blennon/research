__author__ = 'bill'
from brian import *

def pf_rates(t):
    '''
    returns a firing rate as a function of time. To be used with
    the Poisson group

    t: is the simulation time step
    '''
    if 50*ms <= t < 100*ms:
        return 100*Hz
    return 0*Hz