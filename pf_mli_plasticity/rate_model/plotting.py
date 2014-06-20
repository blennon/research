__author__ = 'bill'
from pylab import *

def set_axes():
    xlabel('time (ms)',fontsize=14)
    xticks(fontsize=14)
    yticks(fontsize=14)

def plot_mli(MLI_recording, ls='-', label='MLI Firing rate'):
    plot(MLI_recording[0,:], color='#CC3333', ls=ls, lw=3., label=label)
    legend(loc='upper left', fontsize=14)
    set_axes()
    title('MLI Firing rate', fontsize=16)

def plot_pf(PF_recording, W_recording, ind, c='b'):
    plot(PF_recording[ind,:], color=c, lw=3., label='PF%s firing rate'%ind)
    plot(W_recording[ind,:], color=c, lw=3., ls='--', label='PF%s-MLI weight'%ind)
    legend(loc='upper center', fontsize=14)
    set_axes()
    ylim([0,2])
    title('PF%s Firing rate'%ind, fontsize=16)

def plot_figures(MLI_recording,PF_recording, W_recording):
    figure(figsize=(8,4))
    plot_mli(MLI_recording)
    figure(figsize=(8,4))
    plot_pf(PF_recording, W_recording,0)
    figure(figsize=(8,4))
    plot_pf(PF_recording, W_recording,1,'g')

def plot_figures(MLI_recording,PF_recording, W_recording):
    figure(figsize=(8,4))
    plot_mli(MLI_recording)
    figure(figsize=(8,4))
    plot_pf(PF_recording, W_recording,0)
    figure(figsize=(8,4))
    plot_pf(PF_recording, W_recording,1,'g')

def plot_weight_over_trials(PF_weights_by_trial, ind, c='b', ls='--'):
    plot(PF_weights_by_trial[:,ind], color=c, lw=3., ls=ls, label='PF%s-MLI Weight'%ind)
    legend(loc='lower right', fontsize=14)
    xlabel('Trial',fontsize=14)
    xticks(fontsize=14)
    yticks(fontsize=14)
    title('PF Weights', fontsize=16)