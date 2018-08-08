import pickle
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 18})


def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    See https://gist.github.com/bhawkins/3535131
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1), np.percentile(y, 25., axis=1),  np.percentile(y, 75., axis=1)


def produceLineGraph(stTestValues_filepath, dataset, folder):
    data = pickle.load(open(stTestValues_filepath, 'rb'))

    stTestValues = data[dataset]["vals"]
    names = data[dataset]["names"]

    dataPred = np.concatenate([s[1] for s in stTestValues])
    dataGT = np.concatenate([s[2] for s in stTestValues])
    order = np.argsort(dataGT)

    ypos = list(range(len(order)))

    N = len(order) // 12
    if N % 2 == 0:
        N += 1
    dataDiff = np.abs(dataPred[order] - dataGT[order])
    dataMed, data25, data75 = medfilt(dataDiff, k=N)
    

    f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[1, 4]}, 
                                        figsize=(10,10), sharex=True)
    ax1.hist(dataGT, bins=20, color='g')
    ax1.set_ylabel('Distribution')
    ax1.set_xlim(0, 1.0)
    #ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))

    ax2.fill_betweenx(ypos, dataGT[order] - data75, dataGT[order] + data75, 
                        color='#aac0ff', label="Prediction error\n(75th percentile)")
    ax2.fill_betweenx(ypos, dataGT[order] - dataMed, dataGT[order] + dataMed, 
                        color='#728fff', label="Prediction error\n(50th percentile)")
    ax2.fill_betweenx(ypos, dataGT[order] - data25, dataGT[order] + data25, 
                        color='#0f4ebe', label="Prediction error\n(25th percentile)")
    ax2.scatter([dataGT[order]], ypos, s=7, c='g', label="Expert scores")
    ax2.legend(loc=2, scatterpoints=3)
    ax2.set_xlabel("Quality")
    ax2.set_ylabel("Samples (ordered by score)")
    ax2.set_ylim(0, len(ypos))

    plt.subplots_adjust(hspace=0.05)
    plt.savefig(os.path.join(folder, 'graphdist.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(folder, 'graphdist.png'), bbox_inches='tight')
    plt.savefig(os.path.join(folder, 'graphdist.tiff'), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce the distribution of errors over the dataset')
    parser.add_argument('scores', help="pkl file containing the predictions and scores (usually named 'bestNetworkPredictions.pkl')")
    parser.add_argument('outputfolder', help="Folder where to put the figures")
    parser.add_argument('dataset', help="'train' or 'test' (default : test)", choices=["train", "test"], default="test")
    args = parser.parse_args()

    if not os.path.exists(args.scores):
        print("{} do not exist in the filesystem!".format(args.scores))
        exit()

    produceLineGraph(args.scores, args.dataset, args.outputfolder)
