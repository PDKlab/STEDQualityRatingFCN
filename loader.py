"""
Loader class definition
"""

import os
import random
import warnings
from copy import deepcopy

import numpy as np
from scipy.ndimage import interpolation
from skimage.transform import rotate
import torch
from torch.autograd import Variable

# Ensuring that we always start from the same random state (for reproductibility)
random.seed(42)
np.random.seed(42)

class DatasetLoader:
    """
    Dataset loader helper. This class handles all tasks related to the dataset,
    including loading, whitening, shuffling, and doing data augmentation.

    It acts as a generator and can be iterated on. At each iteration, it will
    return a 4-tuple : (input_images, output_scores, masks, filenames)
    As other generators, it raises a StopIteration exception when the epoch is done.

    It uses a caching mechanism to avoid loading the same data over and over.
    """

    def __init__(self, folder, batchSize, useCuda, 
                    doDataAugmentation=0, maskType="otsu", cacheSize=0, limitData=-1):
        """
        Initialize the loader.

        Parameters:
        * `folder` is the root folder of the dataset, which must contain `train` and `test` subfolders
        * `batchSize` is the maximum size of batches to return
        * `useCuda` indicates if the loader should put the data on the GPU before returning them
        * `doDataAugmentation` is a number indicating whether data augmentation should be performed
                or not. It must be between 0 and 1, and set the data augmentation probability.
                In other words, 0.0 never performs any data augmentation (useful for validation set,
                for instance) while 1.0 always does it.
        * `maskType` is the type of mask to use. Valid values are 'otsu', 'li', and 'nonzero'. Otsu and
                Li are the adaptative threshold algorithms, while 'nonzero' simply create a mask
                encompassing all non zero elements of the input image
        * `cacheSize` sets the maximum size of the cache in MB. Set it to 0 to deactivate caching entirely.
        * `limitData` artificially limits the amount of data in each set. If > 0, its value is used as
                the size of the set, except if the set already has fewer samples. If < 0, the full set
                is used.

        Note that this script assumes that a file named "stats.txt" is available in "datasets_preparation".
        This file should contains a first column indicating the dataset name, followed by a tabulation,
        and by 4 values (mean, std, min, max) about the dataset. This file may have an arbitrary number
        of lines.
        """
        # Assigning member variables
        self.root = folder
        self.bsize = batchSize
        self.cuda = useCuda
        self.maskType = maskType
        self.dataAug = doDataAugmentation
        self.cacheSize = cacheSize * 1024**2        # cacheSize is in MB, we convert it in bytes

        # Opening the statistics folder
        stats = open('datasets_preparation/stats.txt', 'r').read().split("\n")
        for lineS in stats:
            if len(lineS) < 4:
                continue
            # Caution : this line should be changed if the stats of the image data are too far
            # from the phalloidin ones (mean, minimum, maximum, etc.)
            if lineS.strip().split()[0] == "phalloidin":
                self.stats = dict(zip(("mean", "std", "min", "max"), 
                                    [float(s) for s in lineS.strip().split()[1].split(",")]))

        # In some cases, the dataset may contain fixed scores, we load them
        # (this is optional)                           
        self.fixedscores = {}
        if os.path.exists(os.path.join(self.root, "fixedscores.csv")):
            for line in open(os.path.join(self.root, "fixedscores.csv"), 'r'):
                if len(line) < 4:
                    continue
                s = line.strip().split(",")
                self.fixedscores[int(s[0])] = float(s[1])
        
        # The size of the mask is the same as the input images
        self.sizeMask = (224, 224)

        # Initialize the cache
        self.cache = {}

        # Walking through the dataset folder to find sample files
        self.listFiles = []
        for fname in os.listdir(folder):
            if not fname.endswith('.npz'):
                continue
            fnamesplit = fname[:-4].split("-")
            if int(fnamesplit[0]) in self.fixedscores and self.fixedscores[int(fnamesplit[0])] < 0:
                continue
            self.listFiles.append(fname)
        
        # Apply the limitation on the number of samples, if needed
        if limitData > 0 and limitData < len(self.listFiles):
            self.listFiles = random.sample(self.listFiles, limitData)

        # Compute the optimal batch size value
        minBatchNbr = np.ceil(len(self.listFiles) / self.bsize)
        newBatchSize = np.ceil(len(self.listFiles) / minBatchNbr)
        if self.bsize != newBatchSize:
            print("Batch size changed from {} to {} to be even amongst mini-batches".format(self.bsize, newBatchSize))
        self.bsize = int(newBatchSize)

        # Set the position pointer at the beginning of the data
        self.pos = 0

    def newEpoch(self):
        # Shuffle the dataset and reset the position pointer
        random.shuffle(self.listFiles)
        self.pos = 0

    def __next__(self):
        # Produce and return the next batch
        if self.pos >= len(self.listFiles):
            raise StopIteration
        
        X, y, M, names = [], [], [], []
        for i in range(self.bsize):
            if self.pos + i >= len(self.listFiles):
                # We reached the end of the dataset
                break

            fname = self.listFiles[self.pos + i]
            fnamesplit = fname[:-4].split("-")
                
            fpath = os.path.join(self.root, fname)
            if fpath not in self.cache:
                npd = np.load(fpath)
                if self.maskType == "nonzero":
                    d = [npd['arr_0'], (npd['arr_0'] > 0).astype('float32'), fname[:-4]]
                elif self.maskType == "otsu":
                    d = [npd['arr_0'], (npd['arr_2'] > 0).astype('float32'), fname[:-4]]
                elif self.maskType == "li":
                    d = [npd['arr_0'], (npd['arr_3'] > 0).astype('float32'), fname[:-4]]
                else:
                    raise ValueError("Invalid mask type : " + self.maskType)
                d[1] = interpolation.zoom(d[1], (self.sizeMask[0]/d[1].shape[0], self.sizeMask[1]/d[1].shape[1]), order=0, prefilter=False)
                self.cache[fpath] = d
            data = deepcopy(self.cache[fpath])

            # Rescale the data
            data[0] -= self.stats['min']
            data[0] /= 0.8*(self.stats['max'] - self.stats['min'])
            data[0] = np.clip(data[0], 0, 1)

            # Data Augmentation
            if self.dataAug > 0:
                nrot = random.randint(0, 2)
                if random.random() < self.dataAug:
                    # 90 degree rotation
                    data[0] = np.rot90(data[0], nrot, axes=(0,1)).copy()
                    data[1] = np.rot90(data[1], nrot, axes=(0,1)).copy()

                if random.random() < self.dataAug:
                    # Left-right flip
                    data[0] = np.fliplr(data[0]).copy()
                    data[1] = np.fliplr(data[1]).copy()

                if random.random() < self.dataAug:
                    # Intensity scaling
                    intensity_scale = np.clip(np.random.lognormal(0.01, np.sqrt(0.01)), 0.7, 1.4)
                    data[0] = np.clip(data[0]*intensity_scale, 0, 1)

                if random.random() < self.dataAug:
                    # Gamma adaptation
                    gamma = np.clip(np.random.lognormal(0.005, np.sqrt(0.005)), 0.8, 1.2)
                    data[0] = np.clip(data[0]**gamma, 0, 1)

                if random.random() < self.dataAug:
                    # Random arbitrary rotation
                    rotation_amnt = np.random.uniform(0, 360)
                    data[0] = rotate(data[0], rotation_amnt, mode='reflect')
                    data[1] = rotate(data[1].astype('float32'), rotation_amnt, mode='reflect') > 0.5

            if data[0].shape[0] % 32 != 0:
                warnings.warn("Image height must be a multiple of 32! Will crop the bottom of the image to ensure it.")
                data[0] = data[0][:-(data[0].shape[0] % 32)]

            if data[0].shape[1] % 32 != 0:
                warnings.warn("Image width must be a multiple of 32! Will crop the right of the image to ensure it.")
                data[0] = data[0][:, :-(data[0].shape[1] % 32)]
            
            X.append(data[0])
            if int(fnamesplit[0]) in self.fixedscores:
                y.append(self.fixedscores[int(fnamesplit[0])])
            else:
                y.append(float(fnamesplit[1]))
            M.append(data[1])
            names.append(data[2])
        
        # Convert to numpy arrays
        X = np.array(X)
        X = X.reshape(-1, 1, X.shape[1], X.shape[2])
        M = np.array(M)
        y = np.array(y)

        # Convert to Torch tensors (optionally with CUDA)
        Xtorch = Variable(torch.Tensor(X))
        Mtorch = Variable(torch.Tensor(M))
        ytorch = Variable(torch.Tensor(y))
        if self.cuda:
            Xtorch = Xtorch.cuda()
            ytorch = ytorch.cuda()
            Mtorch = Mtorch.cuda()
        
        # Update the position pointer and return the batch
        self.pos += len(y)
        return Xtorch, ytorch, Mtorch, names
    
    def __iter__(self):
        return self

    def __len__(self):
        # The length of a dataset is defined by the number of batches it contains
        return int(np.ceil(len(self.listFiles) / self.bsize))
