# GPU-parallelization
This repo contains Python code for Parallelization strategies used in Machine Learning 

## Setting up python env
Run `./bin/setup_venv` for setting up virtual environment for runnning experimentations.  

## 01_Vanilla_ML.ipynb
I have written this file to recreate what i have learnt in Foundation of AI course. I am using a single GPU to run a classification network. I am using MNIST had-written numbers dataset for training and testing.

Model used is a simple feed-forward classifier network which does not employ any type of  Convolution layer. Model uses a flatten image as input passed in through a neural architecture using three hidden layers. 

Learning rate is kept low, as well as SGD (Stocastic Gradient Decent) is used for back propogation. These choices make sure that the Model reaches convergence very slowly(in a duration of 20 Epochs).

Training time is recorded to be around `382.6086 seconds` and output curves show smooth learning.
