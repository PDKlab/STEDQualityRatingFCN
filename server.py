import os
import argparse
import json
import datetime

import numpy as np

import torch
from torch.autograd import Variable

from flask import Flask
from flask import render_template, request

from skimage import filters

from networks import NetTrueFCN

thresholdType = "otsu"

def get_foreground(img, threshalgo="otsu", sigma=10):
    """Gets the foreground using a gaussian blur of sigma = 10 and the otsu threshold.

    :param img: A 2D numpy array

    :returns : A binary 2D numpy array of the foreground
    """
    global thresholdType
    blurred = filters.gaussian(img, sigma=sigma)
    blurred /= blurred.max()
    if thresholdType == "li":
        val = filters.threshold_li(blurred)
    elif thresholdType == "otsu":
        val = filters.threshold_otsu(blurred)
    else:
        raise ValueError("Invalid threshold type : " + threshalgo)
    return (blurred > val).astype('float32')


app = Flask(__name__)

@app.route("/", methods=['POST'])
def get_score():
    global model, min_, max_, useCuda
    data = json.loads(request.data.decode('utf-8'))
    img = np.array(data['img'], dtype=data['img-type'])

    mask = get_foreground(img)
    img = img[np.newaxis, np.newaxis, :, :] - min_
    img /= 0.8*(max_ - min_)
    img = np.clip(img, 0, 1)
    imgTorch = Variable(torch.Tensor(img.astype('float32')))

    mask = mask[np.newaxis, np.newaxis, :, :]
    maskTorch = Variable(torch.Tensor(mask))

    if useCuda:
        imgTorch = imgTorch.cuda()
        maskTorch = maskTorch.cuda()
    
    heatmap, scoreTorch = model.forward(imgTorch, mask=maskTorch)

    score = float(scoreTorch.cpu().data.numpy().flatten()[0])

    print("Received an image, assigned the score {:.4f}".format(score))
    return json.dumps({'score':score}), 200, {'ContentType':'application/json'}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Server for qualityNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', help='port if using virtual net', type=int, default=5000)
    parser.add_argument('--cuda', help='use GPU or not', action="store_true", default=False)
    parser.add_argument('-v', '--verbose', help='print information', action="store_true", default=False)
    parser.add_argument('modelpath', help='Path to the model to load (.net file)', type=str)
    parser.add_argument('experiment', help='Structure to use (tubulin, PSD95, etc., named exactly as the folder)', type=str)
    parser.add_argument('--maskthreshold', type=str, choices=["otsu", "li"], default="otsu", help="Thresholding algorithm for the mask")
    args = parser.parse_args()
    useCuda = args.cuda

    stats = open('stats.txt', 'r').read().split("\n")
    model = NetTrueFCN()
    if useCuda:
        netParams = torch.load(args.modelpath)
    else:
        # The models were trained on GPU, we have to map them on CPU
        netParams = torch.load(args.modelpath, map_location='cpu')
    model.load_state_dict(netParams)
    if useCuda:
        model = model.cuda()
    model.eval()
    
    for lineS in stats:
        if len(lineS) < 4:
            continue
        if lineS.strip().split()[0] == args.experiment:
            print(lineS.strip().split()[1].split(","))
            stats = dict(zip(("mean", "std", "min", "max"), 
                                [float(s.strip()) for s in lineS.strip().split()[1].split(",")]))
            break
    else:
        print("Invalid experiment!")
        exit()

    min_, max_ = stats['min'], stats['max']
    thresholdType = args.maskthreshold

    app.run(host='0.0.0.0', port=args.port, debug=False, use_reloader=False)

