from pylab import *

# Examples
#PG = PoissonGroup(N_PG, lambda t: stimulus(t,15*Hz,15*Hz,pi*Hz,pi,lambda x: 1+cos(x)))
#PG = PoissonGroup(N_PG, lambda t: stimulus(t,15*Hz,30*Hz,(1/2.)* Hz,0.,triangle_wave))

def stimulus(t, min_stim, amp, freq, phase, waveform):
    return min_stim + amp * waveform(freq*t+phase)

def triangle_wave(t):
    '''
    return a triangle wave with height 1.
    peaks at .5, completes in the interval [0,1]
    '''
    if 0 <= t%1. < .5:
        return 2*(t%1.)
    else:
        return -2*(t%1.) + 2. 
    
def square_wave(t):
    '''
    return a square wave with height 1.
    completes in the interval [0,1]
    '''
    if 0 <= t%1. < .5:
        return 0.
    else:
        return 1. 
