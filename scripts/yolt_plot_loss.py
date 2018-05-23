#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 09:42:00 2017

@author: avanetten
"""

import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate
import pandas as pd
import argparse
import numpy as np
import os
import shutil

###############################################################################
def plot_loss_4col(log_dir, figsize=(8,6), twin_axis=False, 
                 rolling_mean_window=30, plot_file='', dpi=300):

    '''if loss file has 4 columns:
        #in yolt2.c: fprintf(lossfile, "%s,%s,%s,%s\n", "Batch_Num", "BatchSize", "N_Train_Ims", "Loss");
    '''
    
    # ingest to df    
    loss_file = os.path.join(log_dir, 'loss.txt')
    df = pd.read_csv(loss_file, sep=',', header=0)
    
    batch = df['Batch_Num'].values
    loss = df['Loss'].values
    batchsize = df['BatchSize'].values[0]
    N_train_ims = df['N_Train_Ims'].values[0]

    N_seen = batch * batchsize
    epoch = 1.*N_seen / N_train_ims
    
    # ylimit
    ylim = (0, 3*np.std(loss) + np.mean(loss))
    
    # plot
    fig, (ax) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))        
    ax.plot(epoch, loss, color='blue', alpha=0.7,
            linewidth=2, solid_capstyle='round', zorder=2)
    #ax.scatter(epoch, loss, color='cyan', alpha=0.3)
    
    # horizintal line at minumum loss
    ax.axhline(y=np.min(loss), c='orange', alpha=0.3, linestyle='--')

    # filter
    #filt = scipy.signal.medfilt(loss, kernel_size=99)
    #ax.plot(epoch, filt, color='red', linestyle='--')

    # spline
    #filt = scipy.interpolate.UnivariateSpline(epoch, loss)
    #ax.plot(epoch, filt(epoch), color='red', linestyle='--')
    
    # better, just take moving average
    roll_mean = pd.rolling_mean(df['Loss'], window=rolling_mean_window)
    #Series.rolling(window=150,center=False).mean()

    ax.plot(epoch[rolling_mean_window:], roll_mean[rolling_mean_window:], 
            color='red', linestyle='--', alpha=0.85)

    ax.set_ylim(ylim)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    #plt.axis('equal')
    #ax.set_title('YOLT Loss')  
    
    # twin axis?
    if twin_axis:
        ax2 = ax.twiny()
        ax2.plot(batch, loss, color='blue', alpha=0.2)
        ax2.set_xlabel('Batches')
        ax2.set_ylim(ylim)
        plt.suptitle('YOLT Loss')

    else:
        ax.set_title('YOLT Loss')  
        plt.tight_layout()
        
    if len(plot_file) > 0:
        plt.savefig(plot_file, dpi=dpi)
        
    #plt.show()

    return

###############################################################################
def plot_loss_2col(df, figsize=(8,6), batchsize=64, 
                 N_train_ims=2418, twin_axis=False, 
                 rolling_mean_window=100, plot_file='', dpi=300, 
                 verbose=True):
    '''if loss file only has two columns: batch_num and loss'''
    
    batch = df['Batch_Num'].values
    loss = df['Loss'].values
    N_seen = batch * batchsize
    epoch = 1.*N_seen / N_train_ims
    
    # ylimit
    #loss_clip = np.clip(loss, np.percentile(loss, 0.01), np.percentile(loss, 0.98))
    #ymin_plot = max(0,  np.mean(loss_clip) - 2*np.std(loss_clip))
    #ymax_plot = np.mean(loss_clip) + 2*np.std(loss_clip)
    #ylim = (ymin_plot, ymax_plot)
    ylim = (0.9*np.min(loss), np.percentile(loss, 99.5))
    
    if verbose:
        print "batch:", batch
        print "loss:", loss
        print "ylim:", ylim
    
    # plot
    fig, (ax) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))        
    ax.plot(epoch, loss, color='blue', alpha=0.7,
            linewidth=2, solid_capstyle='round', zorder=2)
    #ax.scatter(epoch, loss, color='cyan', alpha=0.3)
    
    # horizintal line at minumum loss
    ax.axhline(y=np.min(loss), c='orange', alpha=0.3, linestyle='--')

    # filter
    #filt = scipy.signal.medfilt(loss, kernel_size=99)
    #ax.plot(epoch, filt, color='red', linestyle='--')

    # spline
    #filt = scipy.interpolate.UnivariateSpline(epoch, loss)
    #ax.plot(epoch, filt(epoch), color='red', linestyle='--')
    
    # better, just take moving average
    #Series.rolling(window=150,center=False).mean()
    roll_mean = df['Loss'].rolling(window=rolling_mean_window, center=False).mean()
    #roll_mean = pd.rolling_mean(df['Loss'], window=rolling_mean_window)
    ax.plot(epoch[int(1.1*rolling_mean_window): ], roll_mean[int(1.1*rolling_mean_window): ], 
            color='red', linestyle='--', alpha=0.85)

    ax.set_ylim(ylim)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(color='gray', alpha=0.4, linestyle='--')
    #plt.axis('equal')
    #ax.set_title('YOLT Loss')  
    
    # twin axis?
    if twin_axis:
        ax2 = ax.twiny()
        ax2.plot(batch, loss, color='blue', alpha=0.2)
        ax2.set_xlabel('Batches')
        ax2.set_ylim(ylim)
        plt.suptitle('YOLT Loss')

    else:
        ax.set_title('YOLT Loss')  
        plt.tight_layout()
        
    if len(plot_file) > 0:
        plt.savefig(plot_file, dpi=dpi)

    return


def main():
    
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('--path', type=str, default='/raid/local/src/yolt2/results/',
    #                    help="path to package")
    parser.add_argument('--res_dir', type=str, default='oops',
                        help="results")
    parser.add_argument('--rolling_mean_window', type=int, default=100,
                        help="Window for rolling mean")
    parser.add_argument('--sep', type=str, default=' ',
                        help="csv separator")
    parser.add_argument('--batchsize', type=int, default=64,
                        help="Training epochs")
    parser.add_argument('--N_train_ims', type=int, default=2418,
                        help="Number of images in training corpus")
    parser.add_argument('--dpi', type=int, default=150,
                        help="dots per inch for plotting")

    args = parser.parse_args()

    # set directories
    #res_dir = os.path.join(args.path, args.res_dir)
    if args.res_dir == 'oops':
        #res_dir = os.get_cwd()
        res_dir = os.path.dirname(os.path.realpath(__file__))

    else:
        res_dir = args.res_dir
        
    #log_dir = os.path.join(res_dir, 'logs')
    log_dir = res_dir #os.path.join(res_dir, 'logs')

    print "res_dir:", res_dir
    print "log_dir:", log_dir


    # set plot name
    plot_file = os.path.join(log_dir, 'loss_plot.png')
    twin_axis=True
    
    loss_file = os.path.join(log_dir, 'loss.txt')
    loss_file_p = os.path.join(log_dir, 'loss_for_plotting.txt')
    
    # copy file because it's probably being actively written to
    #cmd = 'cp ' + loss_file + ' ' + loss_file_p
    #print "copy command:", cmd
    #os.system(cmd)
    shutil.copy2(loss_file, loss_file_p)
    
    # ingest to df
    df_tmp = pd.read_csv(loss_file_p, sep=args.sep).dropna()
    
    if len(df_tmp.columns) == 2:
        # ingest to df
        df = pd.read_csv(loss_file_p, sep=args.sep, names=['Batch_Num', 'Loss']).dropna()

        # plot
        #plot_loss(res_dir, plot_file=plot_file, twin_axis=twin_axis)
        plot_loss_2col(df, batchsize=args.batchsize, N_train_ims=args.N_train_ims, 
                 plot_file=plot_file, twin_axis=twin_axis, 
                 rolling_mean_window=args.rolling_mean_window,
                 dpi=args.dpi)

    else:
        df = pd.read_csv(loss_file_p, sep=args.sep, names=['Batch_Num', 'BatchSize', 'N_Train_Ims', 'Loss']).dropna()
        
        #res_dir = '/Users/avanetten/Documents/cosmiq/yolt2/results/train_cars_0.3m_cfg=ave_13x13_2017_08_12_18-35-25/'    
        plot_loss_4col(log_dir, plot_file=plot_file, twin_axis=twin_axis,
                       dpi=args.dpi)
    
    

if __name__ == "__main__":
    main()
