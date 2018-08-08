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

You can use the `produce_heatmaps.py` script, located in `results_processing`, with the following invocation:

```shell
python produce_heatmaps.py <dataset folder> <network folder> <output folder>
```

Where `dataset_folder` is a folder containing a `test` subfolder with NPZ files (see above), `network folder` a folder containing a `params.net` file and `output folder` the directory where to put the results. By default, the script also outputs colorbar and overall network prediction, but it can also output the sole heatmap by using the `--heatmaponly` flag.

## Using the server

Prior to start the server, a trained model is needed. You can either use a pretrained model amongst the ones provided in `trained_models` or train one yourself (see previous section).

### Starting the server

```shell
docker run --rm -p <PORT>:5000 qnet /bin/bash -c "cd /workspace/executable/ && python server.py trained_models/<model>/params.net <experiment>"
```

Where:

* `<PORT>` is the port you want to use on your machine. You will need to remember this port number to connect the client in the next step.
* `<model>` is the network you want to use (see the folders in `trained_models`). Alternatively, you may also use your own network by providing the path to its parameters checkpoint file.
* `<experiment>` indicates which mean/std.dev to use to standardize the images. It can be *phalloidin* (e.g. Actin), *tubulin*, *CaMKII_Neuron*, *PSD95_Neuron*, *LifeAct_Neuron*, or *Membrane_Neuron*. Alternatively, if you use a new dataset, you may add its relevant statistics in `datasets_preparation/stats.txt` and provide the first field of the line added here.

Also note that you **must** pass the `--cuda` flag if you want the processing to be done on GPU.


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

