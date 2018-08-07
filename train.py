"""
Main training script (see README)
"""

import os
import time
import shutil
import random
import argparse
import pickle
from collections import defaultdict

import matplotlib
# We select a backend that will work on any configuration
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch

import networks
from loader import DatasetLoader


# Helper print function
ptime = None
printlevel = 1
LEVELS = {0 : "DEBUG", 1 : "INFO", 2 : "WARN", 3 : "ERROR", 4 : "FATAL"}
def _print(txt, level=1):
    global ptime
    if level < printlevel:
        return
    if ptime is None:
        delay = 0.0
    else:
        delay = time.time() - ptime

    print("{} [{:.2f}] {}".format(LEVELS[level], delay, txt))
    ptime = time.time()

def load(outputpath):
    # Load a network from a checkpoint folder
    # This folder must contain:
    #   - 'params.net': the parameters of the network to be restored
    #   - 'optimizer.data': the optimizer state
    #   - 'statsCkpt.pkl': the statistics and parameters of the run so far
    netParams = torch.load(os.path.join(outputpath, "params.net"))
    optimizerParams = torch.load(os.path.join(outputpath, "optimizer.data"))
    stats = pickle.load(open(os.path.join(outputpath, "statsCkpt.pkl"), 'rb'))
    return netParams, optimizerParams, stats

def save(outputpath, net, opt, stats):
    # Save network parameters and optimizer state.
    torch.save(net.state_dict(), os.path.join(outputpath, "params.net"))
    torch.save(optimizer.state_dict(), os.path.join(outputpath, "optimizer.data"))
    pickle.dump(stats, open(os.path.join(outputpath, "statsCkpt.pkl"), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('inputfolder',
                        help="Folder containing the dataset (npz files).\nIt must contain train and test subfolders.")
    parser.add_argument('outputfolder', 
                        help="Folder where to put the results")
    parser.add_argument('--cuda', action="store_true", default=False,
                        help='Use CUDA (else, default to CPU mode)')
    parser.add_argument('--batchsize', type=int, default=96,
                        help='Maximum batch size (the exact number will be adjusted depending on the size of the dataset)',)
    parser.add_argument('--maskthreshold', type=str, choices=["otsu", "li", "nonzero"], default="otsu",
                        help="Thresholding algorithm to generate the mask")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Optimizer learning rate")
    parser.add_argument('--lrdecay', type=float, default=100,
                        help="Number of epochs after which the learning rate is halved")
    parser.add_argument('--epochs', type=int, default=350,
                        help='Maximum number of training epochs')
    parser.add_argument("--limitdata", default=-1, type=int, 
                        help="If > 0, limit the number of training samples to this value. If -1 (default), no limit.")
    parser.add_argument("--dataaugrate", default=0.5, type=float, 
                        help="Data augmentation rate, [0, 1]")
    parser.add_argument("--finetuneFrom", type=str, 
                        help="Network parameter file (.net) to finetune from", default="")
    parser.add_argument("--startFromCkpt", action='store_const', const=True, default=False, 
                        help="Restart from the state saved in the output directory")
    parser.add_argument("--overwrite", action='store_const', const=True, default=False, 
                        help="Overwrite the output directory if it already exists, else abort")

    _print("Parsing arguments...")
    args = parser.parse_args()

    _print("Initializing loaders...")
    # Training set
    dataTrain = DatasetLoader(os.path.join(args.inputfolder, 'train'), args.batchsize, args.cuda, 
                                doDataAugmentation=args.dataaugrate, cacheSize=4000, maskType=args.maskthreshold, 
                                limitData=args.limitdata)
    # Training set, but without data augmentation (so to stabilize batch normalization on small datasets)
    dataTrainNoAug = DatasetLoader(os.path.join(args.inputfolder, 'train'), args.batchsize, args.cuda,
                                doDataAugmentation=0.0, cacheSize=4000, maskType=args.maskthreshold, 
                                limitData=args.limitdata)
    # Validation set
    dataTest = DatasetLoader(os.path.join(args.inputfolder, 'test'), args.batchsize, args.cuda,
                                doDataAugmentation=0.0, cacheSize=4000, maskType=args.maskthreshold)

    # Stats object
    stats = defaultdict(list)

    _print("Create network and optimizer")
    if args.startFromCkpt:
        # We restart from an already existing checkpoint
        netParams, optParams, stats = load(args.outputfolder)
        net = networks.NetTrueFCN()
        net.load_state_dict(netParams)
        if args.cuda:
            """
            http://pytorch.org/docs/0.3.1/nn.html#torch.nn.Module.cuda
            This also makes associated parameters and buffers different objects. 
            So it should be called before constructing optimizer if the module will live on GPU while being optimized.
            """
            net = net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        optimizer.load_state_dict(optParams)
        startEpoch = len(stats['train'])
        _print("Restarted from checkpoint")

    elif os.path.exists(args.outputfolder):
        # The output folder already exists, but we want to start a new training
        if args.overwrite:
            _print("Deleting {}".format(args.outputfolder))
            shutil.rmtree(args.outputfolder)
        else:
            _print("Output directory {} already exists, stopping".format(args.outputfolder))
            exit()
    
    if not args.startFromCkpt:
        net = networks.NetTrueFCN()
        if args.finetuneFrom != "":
            # Do we fine-tune from a pretrained network?
            netParams = torch.load(os.path.join(args.finetuneFrom))
            net.load_state_dict(netParams)
            
        if args.cuda:
            """
            http://pytorch.org/docs/0.3.1/nn.html#torch.nn.Module.cuda
            This also makes associated parameters and buffers different objects. 
            So it should be called before constructing optimizer if the module will live on GPU while being optimized.
            """
            net = net.cuda()
        
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        startEpoch = 0
        os.makedirs(args.outputfolder)

    # Ensure that the BN layers momentum is set to a reasonable value
    for child in net.children():
        if type(child) == torch.nn.BatchNorm2d:
            child.momentum = 0.5

    # Set learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lrdecay, gamma=0.5)
    
    _print("Create loss")
    criterion = torch.nn.MSELoss()
    if args.cuda:
        criterion = criterion.cuda()
    minValidLoss = np.inf

    # Training loop
    _print("Start training")
    for epoch in range(startEpoch, args.epochs):
        # Shuffle the train set
        dataTrain.newEpoch()
        dataTest.newEpoch()
        dataTrainNoAug.newEpoch()

        scheduler.step()
        stLossTrain, stLossTest,  = [], []
        stTrainValues, stTrainNames = [], []
        stTestValues, stTestNames = [], []

        # Put the network in train mode
        net.train()
        for (X, y, masks, names) in dataTrain:
            # New batch; we reset the gradients accumulation
            optimizer.zero_grad()
            stTrainNames.extend(names)

            # Prediction and loss computation
            predconv, pred = net.forward(X, mask=masks)
            loss = criterion(pred, y)

            # For logging purposes
            stLossTrain.append(loss.cpu().data.numpy())
            stTrainValues.append((loss.cpu().data.numpy(), 
                            pred.cpu().data.numpy(), 
                            y.cpu().data.numpy()))

            # Back-propagation and optimizer step
            loss.backward()
            optimizer.step()
            
            # Avoid any memory leak
            del X, y, masks, pred, predconv, loss
        
        # The data augmentation can considerably drift the avg/std values of a batch.
        # To avoid any issue in validation (where BN layers use estimates from the
        # training batches), we forward the _non-augmented_ training set again in the
        # network. No need to compute the loss or backpropagate, it is only to adjust
        # the BN layers parameters.
        for (X, y, masks, names) in dataTrainNoAug:
            predconv, pred = net.forward(X, mask=masks)
            del X, y, masks, pred, predconv

        # Evaluation mode
        net.eval()
        for (X, y, masks, names) in dataTest:
            stTestNames.extend(names)

            # Prediction and loss computation
            predconv, pred = net.forward(X, mask=masks)
            loss = criterion(pred, y)

            # For logging purposes
            stLossTest.append((loss.cpu().data.numpy()))
            stTestValues.append((loss.cpu().data.numpy(), 
                            pred.cpu().data.numpy(), 
                            y.cpu().data.numpy()))

            # Avoid any memory leak
            del X, y, masks, pred, predconv, loss
        
        # Aggregate stats
        for key,func in zip(('trainMean', 'trainMed', 'trainMin', 'trainStd'),
                            (np.mean, np.median, np.min, np.std)):
            stats[key].append(np.sqrt(func(stLossTrain)))
        
        for key,func in zip(('testMean', 'testMed', 'testMin', 'testStd'),
                            (np.mean, np.median, np.min, np.std)):
            stats[key].append(np.sqrt(func(stLossTest)))

        # Save the loss curves
        plt.figure(figsize=(12, 9))
        plt.plot(stats['trainMean'], linewidth=2, color='#00CC00', linestyle="solid", label="Train")
        plt.fill_between(np.arange(len(stats['trainMean'])), 
                            np.array(stats['trainMean']) - np.array(stats['trainStd']),
                            np.array(stats['trainMean']) + np.array(stats['trainStd']),
                            color='#99FF99', alpha=0.7)
        plt.plot(stats['testMean'], linewidth=2, color='#0000CC', linestyle="solid", label="Validation")
        plt.fill_between(np.arange(len(stats['testMean'])), 
                            np.array(stats['testMean']) - np.array(stats['testStd']),
                            np.array(stats['testMean']) + np.array(stats['testStd']),
                            color='#9999FF', alpha=0.7)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("RMSE over the predicted scores")
        plt.savefig(os.path.join(args.outputfolder, "loss.png"), bbox_inches='tight')
        plt.close()

        # If the network is the best so far (in validation), we keep it
        isBest = False
        if minValidLoss > stats['testMean'][-1]:
            print("New best network ({} RMSE is better than the previous {})".format(stats['testMean'][-1], minValidLoss))
            minValidLoss = stats['testMean'][-1]
            save(args.outputfolder, net, optimizer, stats)
            isBest = True
            
            pickle.dump({
                "train":{
                        "names": stTrainNames, "vals":stTrainValues
                },
                "test":{
                        "names": stTestNames, "vals":stTestValues
                        }
            }, open(os.path.join(args.outputfolder, 'bestNetworkPredictions.pkl'), 'wb'))

        _print("Epoch {} done!\n\tAvg loss train/validation : {} / {}".format(epoch, stats['trainMean'][-1], stats['testMean'][-1]))

        # Save the current stats
        pickle.dump(stats, open(os.path.join(args.outputfolder, "stats.pkl"), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
