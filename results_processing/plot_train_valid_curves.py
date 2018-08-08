import os
import argparse
import pickle

import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 18})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the training curves (training and validation loss)')
    parser.add_argument('statsfile', help="File containing the stats (.pkl)")
    parser.add_argument('outputfolder', help="Folder where to put the figures")
    args = parser.parse_args()

    os.makedirs(args.outputfolder, exist_ok=True)
    stats = pickle.load(open(args.statsfile, 'rb'))

    plt.figure()
    plt.fill_between(np.arange(len(stats['testMean'])), 
                        np.array(stats['testMean']) - np.array(stats['testStd']),
                        np.array(stats['testMean']) + np.array(stats['testStd']), alpha=0.6)
    plt.fill_between(np.arange(len(stats['trainMean'])), 
                            np.array(stats['trainMean']) - np.array(stats['trainStd']),
                            np.array(stats['trainMean']) + np.array(stats['trainStd']), alpha=0.4)
    plt.plot(stats['testMean'], label="Validation RMSE", linewidth=2)
    plt.plot(stats['trainMean'], label="Training RMSE", linewidth=2)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (RMSE)")

    plt.savefig(os.path.join(args.outputfolder, 'loss.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(args.outputfolder, 'loss.tiff'), bbox_inches='tight')
    plt.savefig(os.path.join(args.outputfolder, 'loss.png'), bbox_inches='tight')
