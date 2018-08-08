import json

import numpy as np
from skimage.io import imread
import requests


class VirtualNet(object):
    r"""Defines a remote network

    **Args**:
        :address (str): address of the network
        :port (int): port of the network, default=5000
    """

    def __init__(self, address, port=5000):
        self.address = address
        self.port = port
        self.url = 'http://{}:{}'.format(self.address, self.port)

    def predict(self, img_mask):
        r"""Predicts the score of an image calling a remote network

        **Args**:
            :img_mask (str/tuple): path of saved numpy arrays or tuple of numpy arrays for image and mask

        .. note::

            The image and mask (img_mask) data must be float between 0 and 1

        **Returns**:
            :score (float): score predicted for the image
        """
        if isinstance(img_mask, str):
            npz = np.load(img_mask)
            img, mask = npz['arr_0'], npz['arr_1']
        elif isinstance(img, tuple) or isinstance(img, list) :
            img, mask = img_mask
        else:
            print(' [!] img must be a str or a numpy array')
        tosend = json.dumps({'img':img.tolist(), 
                             'img-type':'{}'.format(img.dtype),
                             'mask': mask.tolist(),
                             'mask-type':'{}'.format(mask.dtype)})
        r = requests.post(self.url, data=tosend)
        return json.loads(r.text)['score']



