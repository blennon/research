__author__ = 'bill'
from pylab import *

def set_axes():
    '''
    format the axis of figures
    '''
    xlabel('time (ms)',fontsize=14)
    xticks(fontsize=14)
    yticks(fontsize=14)

def plot_mli(MLI_recording, ls='-', label='MLI Firing rate'):
    '''
    plot the MLI recording
    '''
    plot(MLI_recording[0,:], color='r', ls=ls, lw=2., label=label)
    legend(loc='upper right', fontsize=14)
    set_axes()
    title('MLI Firing rate', fontsize=16)

def plot_pf(PF_recording, W_recording, ind, c='b'):
    '''
    plot the parallel fiber 'ind' recording
    '''
    plot(PF_recording[ind,:], color=c, lw=2., label='PF%s firing rate'%(ind+1))
    plot(W_recording[ind,:], color=c, lw=2., ls='--', label='PF%s-MLI weight (Trial 1)'%(ind+1))
    legend(loc='upper center', fontsize=14)
    set_axes()
    ylim([0,2])
    title('PF%s Firing rate'%ind, fontsize=16)

def plot_weight_over_trials(PF_weights_by_trial, ind, c='b', legend_on=True):
    '''
    plot the weight values for PF of index 'ind' over trials
    '''
    plot(PF_weights_by_trial[:,ind], color=c, lw=2., label='PF%s-MLI Weight'%(ind+1))
    if legend_on:
        legend(loc='lower right', fontsize=14)
    xlabel('Trial',fontsize=14)
    xticks(fontsize=14)
    yticks(fontsize=14)
    title('PF Weights', fontsize=16)

def plot_weights(pf_weights_by_trial):
    '''
    plot both PF weights over the trials
    '''
    plot_weight_over_trials(pf_weights_by_trial, 0, c='b')
    plot_weight_over_trials(pf_weights_by_trial, 1, c='g')

def plot_dashboard(recordings):
    '''
    plot a dashboard of the activity of PFs, MLI, and weights
    '''
    fig = figure(figsize=(16,10))

    subplot(221)
    plot_mli(recordings[0]['MLI'], ls='--', label='Trial 1')
    plot_mli(recordings[-1]['MLI'], label='Trial 100')

    subplot(222)
    pf_weights_by_trial = squeeze([recordings[i]['W'][:,-1] for i in range(100)])
    plot_weights(pf_weights_by_trial)

    subplot(223)
    plot_pf(recordings[0]['PF'], recordings[0]['W'],0)

    subplot(224)
    plot_pf(recordings[0]['PF'], recordings[0]['W'],1,'g')