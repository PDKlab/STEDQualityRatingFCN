import os
import time
import shutil
import random
import argparse
import pickle
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters
from scipy.misc import imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append("..")
import networks
from loader import DatasetLoader


def load(outputpath):
    netParams = torch.load(os.path.join(outputpath, "params.net"))
    return netParams

def createNet():
    return networks.NetTrueFCN()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test script')
    parser.add_argument('inputfolder', help="Folder containing train and test subfolders")
    parser.add_argument('netfolder', help="Folder containing the network parameters")
    parser.add_argument('outputfolder', help="Folder where to put the figures")
    parser.add_argument('--cuda', help='Use CUDA', action="store_true", default=False)
    parser.add_argument('--batchsize', help='Batch size', type=int, default=64)
    parser.add_argument('--outputsize', type=int, default=224, help="Size of the output (only for the FCN networks), in pixels")
    parser.add_argument('--maskthreshold', type=str, choices=["otsu", "li", "nonzero"], default="otsu", help="Thresholding algorithm for the mask")
    parser.add_argument('--heatmaponly', help="Display the heatmap and only the heatmap", action="store_true", default=False)
    args = parser.parse_args()

    dataTest = DatasetLoader(os.path.join(args.inputfolder, 'test'), args.batchsize, args.cuda,
                                doDataAugmentation=0.0, cacheSize=4000, maskType=args.maskthreshold)
    os.makedirs(args.outputfolder, exist_ok=True)

    print("Load network")
    netParams = load(args.netfolder)
    net = createNet()
    net.load_state_dict(netParams)
    if args.cuda:
        net = net.cuda()
    net.eval()
    
    print("Create loss")
    criterion = nn.MSELoss()
    if args.cuda:
        criterion = criterion.cuda()

    dataTest.newEpoch()
    stLossTest = []
    for batch in dataTest:
        X, y, M, names = batch

        predconv, pred = net.forward(X, mask=M)
        loss = criterion(pred, y)
        stLossTest.append((loss.cpu().data.numpy(), 
                            pred.cpu().data.numpy(), 
                            y.cpu().data.numpy()))

        heatmaps = predconv.cpu().data.numpy()
        sources = X.cpu().data.numpy()

        for i in range(len(names)):
            imHeatmap = heatmaps[i, 0]
            imSource = sources[i, 0]

            imFilteredHeatmap = filters.gaussian(imHeatmap, sigma=5)
            imFilteredHeatmap = imFilteredHeatmap / imFilteredHeatmap.max() * imHeatmap.max()

            if args.heatmaponly:
                fig = plt.figure()
                plt.imshow(imSource, cmap=plt.cm.gray, interpolation='nearest')
                plt.imshow(imFilteredHeatmap, cmap=plt.cm.viridis, interpolation='bilinear', alpha=0.42, vmin=0.1, vmax=0.9)

                imsave(os.path.join(args.outputfolder, names[i] + "-prediction{:.4f}-input.png".format(pred[i].cpu().data.numpy()[0])), imSource)
                
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(os.path.join(args.outputfolder, names[i] + "-prediction{:.4f}.png".format(pred[i].cpu().data.numpy()[0])), bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                plt.figure(figsize=(11, 9))
                plt.imshow(imSource, cmap=plt.cm.gray, interpolation='nearest')
                plt.imshow(imFilteredHeatmap, cmap=plt.cm.viridis, interpolation='bilinear', alpha=0.5, vmin=0.1, vmax=0.9)
                plt.colorbar()
                plt.title("[{}] Score : {:.4f} / Prediction : {:.4f}".format(names[i].split("-")[0], y[i].cpu().data.numpy()[0], pred[i].cpu().data.numpy()[0]))
                #plt.show()
                plt.savefig(os.path.join(args.outputfolder, names[i] + ".png"), bbox_inches='tight')
                plt.close()

        del X, y, pred, predconv, loss
