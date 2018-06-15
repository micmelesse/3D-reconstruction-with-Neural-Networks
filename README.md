# 3D reconstruction with Neural Networks
This is work that I did as part of my [Senior thesis](./3D_reconstruction_with_neural_networks.pdf) at Princeton University. It is an implementation in Tensorflow of the network described by Choy et al in [3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction](https://arxiv.org/pdf/1604.00449.pdf). The project is a neural network capable of performing 3D reconstruction using a variable number of images.

# Demonstration
See the video below to see the network in action over a period of 40 epochs. The more red a voxel is the more certain the network is of its prediction at that position.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/iI6ZMST8Ri0/0.jpg)](https://www.youtube.com/watch?v=iI6ZMST8Ri0)

# Getting Started
The network was trainied using an AWS EC2 [p2.xlarge](https://aws.amazon.com/ec2/instance-types/p2/) instance.
## Prerequisite 

The projects make use of the several python pacakges. It is possible install these package using pip. For example to install the numpy package use the following command.
```
pip install numpy
```
It is possible to install tensorflow using pip as shown above but if you are having issues installing tensorflow on your specific platform follow the instructions [here](https://www.tensorflow.org/install/).
## Setup
To start training the network, one must first setup the network using a shell script. This creates folders for the data, models and a JSON file to store the parameters of the network being trained.
```
sh scripts/setup_dir.sh
```

## Training
You can use preexisting shell scripts to start training the network after setting up the directory. 
```
sh scripts/train.sh
```

