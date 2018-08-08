 Automated STED images quality rating 
========
This repo contains the neural network architecture for predicting the quality score of a STED image. This is the FCN architecture presented used in the paper [Toward intelligent nanoscopy: A machine learning approach for automated optimization of super-resolution optical microscopy](http://tdb).

It contains the code to:

* Train a new model from scratch
* Fine-tune an existing model on a new dataset
* Produce results and heatmaps, as presented in the paper
* Deploy a server able to assess the quality of a STED image through a JSON API


## Installation Instructions

### Installing docker

- Getting the Docker installation file from [https://www.docker.com/community-edition#/download](https://www.docker.com/community-edition#/download) depending of your operating system. 
- Starting the installation and following instruction. It should be straightforward, though docker can take few minutes to start.

### Building the image

If you have access to a nVidia GPU on a Linux computer, we strongly recommend to use it to speed up training and inference. In this case, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) must also be installed. Else, you can fall back to a CPU image, which should have decent inference throughput but require a few hours to train a new model (instead of a few minutes).

- Clone the repository: `git clone https://github.com/PDKlab/STEDQualityRatingFCN`
- Select the right Docker image to build (GPU or CPU): `cd STEDQualityRatingFCN/Dockerfiles/cpu` or `cd STEDQualityRatingFCN/Dockerfiles/gpu`
- Build the Docker image: `docker build -t qnet .`
- You can now start the image using `(nvidia-)docker run -it -v <path to your data>:/mnt/data qnet /bin/bash`

### Alternative installation

You can alternatively use [Anaconda Distribution](https://www.anaconda.com/download) to install all Python requirements. In this case, the following libraries must also be install through `conda install`:

- Numpy and scipy
- Scikit-image
- Matplotlib
- PyTorch 0.3.1 (**not** 0.4 or higher!)


## Train or fine-tune a new model

### Select and prepare the dataset

To train or fine-tune a network, you must have access to a dataset in the right format. The ones used in the paper are publicly available [here](http://tbd). You may also create your own dataset following these guidelines:

* The top directory must contain at least two subfolders: `train` and `test`
* Each of these directories must contain Numpy files (saved with `numpy.savez`, usually with *npz* extension). The *name* of these files must be of the form `X-Y.npz`, where *X* is a unique number among the dataset and Y the quality score, as given by an expert (in the [0, 1] range, at least with 2 digits). These files should *contain* three arrays: the first one being the images themselves, the second one a foregound mask computed using the Li thresholding method, and the third one a foreground mask computed using the Otsu method. Li and Otsu operators can be found in scikit-image *filters* module.
* The average, standard deviation, minimum and maximum of the dataset shall be computed and put in `datasets_preparation/stats.txt` on a line beginning by the name of the top directory.


### Start a new training

To start a new training (initialized with random weights), use the following:

```shell
python train.py <dataset folder> <output folder>
```

The `train.py` script has many more options, see the help form more details (`python train.py --help`).

### Finetune from an existing network

This is the same procedure as the previous one, but it initializes the network with pretrained weights:

```shell
python train.py <dataset folder> <output folder> --finetuneFrom=<path to a params.net file containing the pretrained weights>
```

### Produce heatmaps for existing images

TODO

## Using the server



### Starting the server

You have two choices to start the server:

1. Start with pre-installed trained models in the image:

    `docker run --rm -p 5000:5000 qnet /bin/bash -c "cd /workspace/executable/ && python server.py trained_models/<experiment> <name>"`

   - `<experiment>` is one of the three pre-installed experiment folder in `trained_models`.
        - `alltrained` : the network has seen actin and tubulin proteins during training.
        - `tubulin`: trained only on tubulin.
        - `CaMKII_PSD95_Neuron`: trained on live imaging with CaMKII and PSD95 proteins.
   - `<name>` is the identifier after `-` in the network's name. ex: in `best_model.t7-0`, `<name>`=`0`.

2. Start with your experiment, create a docker volume linking to your experiment folder: 

   `docker run --rm -v "<my-experiment-folder>:/mnt/experiment" -p 5000:5000 qnet /bin/bash -c "cd /workspace/executable/ && python server.py /mnt/experiment <name>"`

   - `<name>` is still the identifier after `-` in the network's name. ex: in `best_model.t7-0`, `<name>`=`0`

### Test with the client

To test with the client: `docker run --rm qnet /bin/bash -c "cd /workspace/executable/ && python client.py --url <server ip> test_samples/<protein>/<test_sample>" `

- `<server ip>:` **on the server-side** can be found with `ifconfig` on unix-like system and `ipconfig` on windows
- `<protein>:` phalloidin or tubulin
- `<test_sample>`: choose one of the test samples

### Using the virtual API

Once a server is started, you can talk to it with the virtual `VirtualNet `API .`src/models/virtual.py` provides an interface example for interacting with the deployed model. It can be used in the following way:

* Instanciate a `VirtualNet`with an ip address and you can use its `.predict(img_mask)` method.
* `img_mask` should be a tuple of 2 arrays `(img, mask)` with values between `0-1`.



## Question?

Feel free to send me an email @ marc-andre.gardner.1 ([a t]) ulaval.ca

