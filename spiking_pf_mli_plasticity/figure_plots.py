__author__ = 'bill'
import sys
sys.path.append('../')
from neuron_models import *
from pylab import *
import seaborn as sns

@check_units(end_time=second)
def plot_pf_mli_w_fig(MLI_V,MLI_S,MLI_R,GR_R,W,fig,defaultclock,xmax=None):
    '''
    :param MLI_V: MLI voltage monitor
    :param MLI_S: MLI spike monitor
    :param MLI_R: MLI rate monitor
    :param GR_R: GR rate monitor
    :param W: synapse weights state monitor
    :param fig: a matplotlib figure object
    :param end_time:
    :return: None
    '''
    sns.set_style('white')
    sns.set_context('paper')
    end_time = float(defaultclock.t)
    if xmax is None:
        xmax = end_time
    T = linspace(0,end_time,1000*end_time)

    gs = GridSpec(3,1,height_ratios=[2,1,3])

    # plot MLI membrane potential trace
    ax1 = fig.add_subplot(gs[0])
    MLI_V.insert_spikes(MLI_S)
    ax1.plot(T,squeeze(MLI_V.getvalues())*1000, color='#8C2318')
    ax1.set_xlim(xmax=xmax)
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel('MLI Membrane Potential (mV)', fontsize=16)
    simpleaxis(ax1)

    # plot MLI firing rate trace
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(T,MLI_R.getvalues(), color='#8C2318', linewidth=1.5)
    ax2.set_xlim(xmax=xmax)
    ax2.tick_params(labelsize=16)
    ax2.set_ylabel('MLI Firing Rate (Hz)', fontsize=16)
    simpleaxis(ax2)

    # Overlay plot of GR firing rate and synapse weight
    ax3 = fig.add_subplot(gs[2])
    gr_frs = squeeze(GR_R.getvalues())
    ax3.plot(T,gr_frs, color='#0B486B')
    ax3.set_ylabel('Superimposed GR Firing Rates (Hz)', color='#0B486B', fontsize=20)
    ax3.tick_params(labelsize=16)
    ax3.spines['top'].set_visible(False)

    ax4 = ax3.twinx()
    ax4.plot(T,mean(W.getvalues(),axis=0), color='#0A835B', linewidth=4.)
    ax4.set_ylabel('Mean Synaptic Weight', color='#0A835B', fontsize=20)
    ax4.tick_params(labelsize=16)
    ax4.set_xlim(xmax=end_time)
    ax3.set_xlabel('time (s)', fontsize=20)
    ax3.set_xlim(xmax=xmax)

    return ax1, ax2, ax3, ax4

def plot_weights_by_trial(fig, W, trial_len, trial_start_time, dt):
    '''
    Plot the value of the weight at the end of each trial. Also show
    the starting weight value as the 0th trial (i.e. during equilibrium
    period).

    fig: matplotlib figure object
    W: weight monitor object
    trial_len: duration of each trial in seconds
    trial_start_time: time when the first
    '''
    # style
    sns.set_style("whitegrid")
    sns.set_context("poster")
    ax = fig.add_subplot(111)

    start_ind = int(trial_start_time/dt)
    # use the first value of each trial as the value at end of previous trial
    wt = W.getvalues()[:,start_ind:][:,::int(trial_len/dt)]
    wt = hstack((wt,W.getvalues()[:,-1][...,None])) # this adds the last value of the last trial to be the end value of last trial
    trials = range(wt.shape[1])
    sns.tsplot(wt, time=trials, ci=100, color='#0A835B',ax=ax, linewidth=3)
    ax.set_ylabel('Synaptic Weight', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_xticks(trials)