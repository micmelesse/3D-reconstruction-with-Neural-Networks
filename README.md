# 3D reconstruction with Neural Networks
This is work that I did as part of my [Senior thesis](./3D_reconstruction_with_neural_networks.pdf) at Princeton University. It is an implementation in Tensorflow of the network described by Choy et al in [3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction](https://arxiv.org/pdf/1604.00449.pdf). The project is a neural network capable of performing 3D reconstruction using a variable number of images.

# Demonstration
See the video below to see the network in action over a period over 40 epochs. The more red a voxel is the more sure the network is of it prediction in that specfic location.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/iI6ZMST8Ri0/0.jpg)](https://www.youtube.com/watch?v=iI6ZMST8Ri0)

# Getting Started

The network was trainied using an AWS EC2 [p2.xlarge](https://aws.amazon.com/ec2/instance-types/p2/) instance.
## prerequisite 
To install tensorflow on your specific platform follow the instructions [here](https://www.tensorflow.org/install/)

## Training
To start training the network use the following command
```
python run.py
```

In addition to that you can use preexisting shell scripts to train the network on macOS or Linux platforms.
```
sh scripts/train.sh
```

