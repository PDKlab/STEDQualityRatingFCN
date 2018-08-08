import os
import json
import argparse

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import filters
import requests

from src.dataset.utils import img_to_float


CONF_PATH = 'Confocal1'
STED_PATH = 'STED'


def get_foreground(img):
    val = filters.threshold_otsu(img)
    return img > val


class VirtualNet(object):
    """This class implements a remote network

    :param address (str): Address of the network.
    :param port (int): Port of the network (default: 5000)
    """

    def __init__(self, address, port=5000):
        self.address = address
        self.port = port
        self.url = 'http://{}:{}'.format(self.address, self.port)

    def predict(self, img_mask):
        """Predict the quality score of an image using a remote neural network.

        :param img_mask (str/tuple): Path of saved numpy arrays or tuple of numpy
                                     arrays for image and mask

        .. note::

            The data type of image and mask contained (or path given) in `img_mask`
            data must be [0, 1] floats.

        :returns: Predicted quality score (float) in [0, 1].
        """
        if isinstance(img_mask, str):
            ext = os.path.splitext(img_mask)[-1]
            if ext == '.npz':
                npz = np.load(img_mask)
                img, mask = npz['arr_0'], npz['arr_1']
            elif ext == '.tiff':
                sted_path = os.path.join(STED_PATH, img_mask)
                conf_path = os.path.join(CONF_PATH, img_mask)
                ## load image
                sted = img_to_float(imread(sted_path))
                conf = img_to_float(imread(conf_path))
                ## generate mask
                fg_c = get_foreground(conf)
                fg_s = fg_c * get_foreground(sted)
                mask = resize(fg_s, (14, 14))
                mask = (mask / numpy.sum(mask)) if numpy.sum(mask) > 0 else mask
                img = sted
            else:
                raise ValueError(f' [!]: the file type {ext} is not known')
        elif isinstance(img_mask, tuple) or isinstance(img_mask, list) :
            img, mask = img_mask
        else:
            print(' [!] img must be a str or a numpy array')
        tosend = json.dumps({'img':img.tolist(),
                             'img-type':'{}'.format(img.dtype),
                             'mask': mask.tolist(),
                             'mask-type':'{}'.format(mask.dtype)})
        r = requests.post(self.url, data=tosend)
        print("VirtualNet call result", r.text)
        return json.loads(r.text)['score']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Client for remote qualityNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--url', help='url if using virtual net', type=str, default='0.0.0.0')
    parser.add_argument('--port', help='port if using virtual net', type=int, default=5000)
    parser.add_argument('--cuda', help='use GPU or not', action="store_true", default=False)
    parser.add_argument('-v', '--verbose', help='print information', action="store_true", default=False)
    parser.add_argument('img', help='path of an image', type=str)
    args = parser.parse_args()

    model = VirtualNet(args.url, args.port)
    score = model.predict(args.img)
    print(' [-] prediction: {:.3f}'.format(score))


