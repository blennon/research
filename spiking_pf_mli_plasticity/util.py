__author__ = 'bill'
from brian import *
import os, errno
import cPickle
import datetime

def mkdir_p(path):
    '''
    recursively create directories.
    
    credit: tzot, Craig Ringer
    from: http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def pickle_results(objects_dict, exper_dir):
    '''
    objects_dict: a dictionary of (object string, object) where the object contains data
    exper_dir: the top-level directory to save results for this experiment
    '''
    today = datetime.datetime.today()
    out_dir = exper_dir+today.strftime("%m-%d-%Y_%H:%M")+'/'
    mkdir_p(out_dir)
    for s,obj in objects_dict.iteritems():
        if '_S' in s:
            with open(out_dir+'%s.pkl'%s,'w') as f:
                cPickle.dump(obj.spiketimes, f)
        else:
            with open(out_dir+'%s.pkl'%s,'w') as f:
                cPickle.dump(obj.getvalues(), f)

def pf_rates(t):
    '''
    returns a firing rate as a function of time. To be used with
    the Poisson group

    t: is the simulation time step
    '''
    if 50*ms <= t < 100*ms:
        return 100*Hz
    return 0*Hz