# CNN-ResNet_Human_sign_language
Image classification using CNN (ResNet50) for speech/hearing-impaired

## Tools Used:
Python, Numpy, TensorFlow (Backend), Keras, matplotlib


## Residual Networks
In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by He et al., allow you to train much deeper networks than were previously practically feasible.

## Goals:

### 1. Implement the basic building blocks of ResNets.
### 2. Put together these building blocks to implement and train a state-of-the-art neural network for image classification.


# MODEL ARCHITECTURE:

The details of this ResNet-50 model are:

Zero-padding pads the input with a pad of (3,3)
## Stage 1:
The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".

BatchNorm is applied to the 'channels' axis of the input.

MaxPooling uses a (3,3) window and a (2,2) stride.


## Stage 2:
The convolutional block uses three sets of filters of size [64,64,256]

The 2 identity blocks use three sets of filters of size [64,64,256]


## Stage 3:
The convolutional block uses three sets of filters of size [128,128,512]

The 3 identity blocks use three sets of filters of size [128,128,512]


## Stage 4:
The convolutional block uses three sets of filters of size [256, 256, 1024]

The 5 identity blocks use three sets of filters of size [256, 256, 1024]


## Stage 5:
The convolutional block uses three sets of filters of size [512, 512, 2048]

The 2 identity blocks use three sets of filters of size [512, 512, 2048]

The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".

The 'flatten' layer doesn't have any hyperparameters or name.

The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be 'fc' + str(classes).
