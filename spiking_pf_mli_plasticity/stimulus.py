__author__ = 'bill'
from brian import *

@check_units(mean_time=second)
def switch_prob(mean_time, dt):
    '''
    compute the probability of switching states (on to off or off to on)
    given the desired mean time (in ms) to stay in the state. This calculation
    is done assuming the times between switching states are generated by a Poisson
    process where ther rate parameter depends on whether the neuron is in the down
    state or up state.

    mean_time: mean time (in ms) to stay in state
    dt: smallest simulation time step that the states are updated on

    returns probability of switching states needed to ensure mean time of
    'mean_time' in state. This should be compared to a pseudo-random number
    drawn from a uniform distribution every dt.
    '''
    n = int(mean_time/dt)
    return 1 - exp(-(1./n))

def switch_states(states, prob_on=.045, prob_off=.035):
    '''
    Implements a Markov process with two states: down-state and up-state, 0 and 1, resp.
    Calling this function switches the states of each element of the vector randomly
    according to prob_on, prob_off.

    states: a binary array describing the state of a neuron (down or up)
    prob_on: probability of switching the state from down to up
    prob_off: probability of switching the state from up to down
    '''
    r = rand(states.shape[0])
    off_to_on_idcs = (states==0) & (r<prob_on)
    on_to_off_idcs = (states==1) & (r<prob_off)
    states[off_to_on_idcs] = 1 # switch off to on
    states[on_to_off_idcs] = 0 # switch on to off
    return states

@check_units(T=second, down_rate=hertz, up_rate=hertz)
def generate_trial_pf_rates(T, N_GR, down_rate, up_rate, p_down_to_up, p_up_to_down):
    '''
    Creates a sequence of GR states based on state transition probabilities. GR rates
    are generated every millisecond.

    T: time duration to generate rates for
    N_GR: number of granule cells
    down_rate: firing rate (Hz) of GRs in their down-state
    p_down_to_up: probability of transitioning from the down state to the up state
    p_up_to_down: probability of transitioning from the up state to the down state

    returns a list of arrays of GR firing rates
    '''
    GR = zeros(N_GR)
    states = []
    for _ in range(int(T/msecond)):
        switch_states(GR, p_down_to_up, p_up_to_down)
        GR_firing_rates = ones(N_GR)*down_rate
        GR_firing_rates[GR>0] = up_rate
        states.append(GR_firing_rates)
    return states

class GR_rates:
    '''
    This class that acts like a function, memorizes a sequence of GR states
    and returns the correct one based on a time step argument.
    '''
    @check_units(T_trial=second,T_CS=second,D_CS=second,dt=second,down_rate=hertz,up_rate=hertz)
    def __init__(self,T_trial,T_CS,D_CS,dt,N_GR,down_rate,up_rate,p_down_to_up,p_up_to_down):
        '''
        T_trial: duration of the trial
        T_CS: start time of conditioned stimulus
        D_CS: duration of conditioned stimulus
        dt: simulation time step
        N_GR: number of granule cells
        down_rate: baseline firing rate of GRs (in their "down" state)
        up_rate: maximum firing rate of GRs (in their "up" state)
        p_down_to_up: probability of transitioning from down to up state
        p_up_to_down: similarly
        '''
        self.T_trial, self.T_CS, self.D_CS = T_trial, T_CS, D_CS
        self.N_GR = N_GR
        self.down_rate = down_rate
        self.CS_cache = generate_trial_pf_rates(D_CS,N_GR,down_rate,up_rate,p_down_to_up,p_up_to_down)
        self.idx = 0
        self.sub_ms_counter = 0
        self.steps_per_ms = int((1*ms)/dt)
        self.calls = []

    def __call__(self, t):
        if self.T_CS <= (t % self.T_trial) < self.T_CS + self.D_CS:
            gr_rate = self.CS_cache[self.idx]
            self.tick()
            return gr_rate
        return ones(self.N_GR)*self.down_rate

    def tick(self):
        '''
        increment the counters for accurate indexing of CS_cache
        '''
        self.calls.append((self.sub_ms_counter, self.idx))
        self.sub_ms_counter += 1
        if self.sub_ms_counter == self.steps_per_ms:
            self.sub_ms_counter = 0
            self.idx = (self.idx + 1) % len(self.CS_cache)