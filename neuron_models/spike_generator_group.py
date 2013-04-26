from brian import *
from brian.directcontrol import FastSpikeGeneratorThreshold
import random as pyrandom
from numpy import where, array, zeros, ones, inf, nonzero, tile, sum, isscalar,\
                  cumsum, hstack, bincount,  ceil, ndarray, ascontiguousarray
from operator import itemgetter
import numpy
from numpy.random import exponential, randint, binomial
from itertools import izip

class SpikeGeneratorGroupDelay(NeuronGroup):
    """
    Same as SpikeGeneratorGroup, I just added **kwargs for testing purposes
    
    Emits spikes at given times
    
    Initialised as::
    
        SpikeGeneratorGroup(N,spiketimes[,clock[,period]])
    
    with arguments:
    
    ``N``
        The number of neurons in the group.
    ``spiketimes``
        An object specifying which neurons should fire and when. It can be a container
        such as a ``list``, containing tuples ``(i,t)`` meaning neuron ``i`` fires at
        time ``t``, or a callable object which returns such a container (which
        allows you to use generator objects even though this is slower, see below). ``i`` can be an integer
        or an array (list of neurons that spike at the same time).
        If ``spiketimes`` is not a list or tuple, the pairs ``(i,t)`` need to be
        sorted in time. You can also pass a numpy array
        ``spiketimes`` where the first column of the array
        is the neuron indices, and the second column is the times in
        seconds. Alternatively you can pass a tuple with two arrays, the first one being the neuron indices and the second one times. WARNING: units are not checked in this case, the time array should be in seconds.
    ``clock``
        An optional clock to update with (omit to use the default clock).
    ``period``
        Optionally makes the spikes recur periodically with the given
        period. Note that iterator objects cannot be used as the ``spikelist``
        with a period as they cannot be reinitialised.
    ``gather=False``
        Set to True if you want to gather spike events that fall in the same
        timestep. (Deprecated since Brian 1.3.1)
    ``sort=True``
        Set to False if your spike events are already sorted.
    
    Has an attribute:
    
    ``spiketimes``
        This can be used to reset the list of spike times, however the values of
        ``N``, ``clock`` and ``period`` cannot be changed. 
        
    **Sample usages**
    
    The simplest usage would be a list of pairs ``(i,t)``::
    
        spiketimes = [(0,1*ms), (1,2*ms)]
        SpikeGeneratorGroup(N,spiketimes)
    
    A more complicated example would be to pass a generator::

        import random
        def nextspike():
            nexttime = random.uniform(0*ms,10*ms)
            while True:
                yield (random.randint(0,9),nexttime)
                nexttime = nexttime + random.uniform(0*ms,10*ms)
        P = SpikeGeneratorGroup(10,nextspike())
    
    This would give a neuron group ``P`` with 10 neurons, where a random one
    of the neurons fires at an average rate of one every 5ms. Please note that as of 1.3.1, this behavior is preserved but will run slower than initializing with arrays, or lists.
    
    **Notes**
    
    Note that if a neuron fires more than one spike in a given interval ``dt``, additional
    spikes will be discarded. A warning will be issued if this
    is detected.

    Also, if you want to use a SpikeGeneratorGroup with many spikes and/or neurons, please use an initialization with arrays.
    
    Also note that if you pass a generator, then reinitialising the group will not have the
    expected effect because a generator object cannot be reinitialised. Instead, you should
    pass a callable object which returns a generator. In the example above, that would be
    done by calling::
    
        P = SpikeGeneratorGroup(10,nextspike)
        
    Whenever P is reinitialised, it will call ``nextspike()`` to create the required spike
    container.
    """
    def __init__(self, N, spiketimes, clock=None, period=None, 
                 sort=True, gather=None, **kwargs):
        clock = guess_clock(clock)
        self.N = N
        self.period = period
        if gather:
            log_warn('brian.SpikeGeneratorGroup', 'SpikeGeneratorGroup\'s gather keyword use is deprecated')
        fallback = False # fall back on old SpikeGeneratorThreshold or not
        if isinstance(spiketimes, list):
            # spiketimes is a list of (i,t)
            if len(spiketimes):
                idx, times = zip(*spiketimes)
            else:
                idx, times = [], []
            # the following try ... handles the case where spiketimes has index arrays
            # e.g spiketimes = [([0, 1], 0 * msecond), ([0, 1, 2], 2 * msecond)]
            # Notes:
            # - if there is always the same number of indices by array, its simple, it's just a matter of flattening
            # - if not, then it requires a for loop, and it's done in the except
            try:
                idx = array(idx, dtype = float)
                times = array(times, dtype = float)
                if idx.ndim > 1:
                    # simple case
                    times = tile(times.reshape((len(times), 1)), (idx.shape[1], 1)).flatten()
                    idx = idx.flatten()
            except ValueError:
                new_idx = []
                new_times = []
                for k, item in enumerate(idx):
                    if isinstance(item, list):
                        new_idx += item # append indices
                        new_times += [times[k]]*len(item)
                    else:
                        new_times += [times[k]]
                        new_idx += [item]
                idx = array(new_idx, dtype  = float)
                times = new_times
                times = array(times, dtype = float)
        elif isinstance(spiketimes, tuple):
            # spike times is a tuple with idx, times in arrays
            idx = spiketimes[0]
            times = spiketimes[1]
        elif isinstance(spiketimes, ndarray):
            # spiketimes is a ndarray, with first col is index and second time
            idx = spiketimes[:,0]
            times = spiketimes[:,1]
        else:
            log_warn('brian.SpikeGeneratorGroup', 'Using (slow) threshold because spiketimes is assumed to be a generator/iterator')
            # spiketimes is a callable object, so falling back on old SpikeGeneratorThreshold
            fallback = True

        if not fallback:
            thresh = FastSpikeGeneratorThreshold(N, idx, times, dt=clock.dt, period=period)
        else:
            thresh = SpikeGeneratorThreshold(N, spiketimes, period=period, sort=sort)
        
        if not hasattr(self, '_initialized'):
            NeuronGroup.__init__(self, N, model=LazyStateUpdater(), threshold=thresh, clock=clock, **kwargs)
            self._initialized = True
        else:
            self._threshold = thresh
 
    def reinit(self):
        super(SpikeGeneratorGroup, self).reinit()
        self._threshold.reinit()
        
    def get_spiketimes(self):
        return self._threshold.spiketimes
    
    def set_spiketimes(self, values):
        self.__init__(self.N, values, period = self.period)
    
    # changed due to the 2.5 issue
    spiketimes = property(get_spiketimes, set_spiketimes)

if __name__== "__main__":
    G = SpikeGeneratorGroupDelay(1,[(0,1*ms)])
    