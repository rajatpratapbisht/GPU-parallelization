# GPU-parallelization
This repo contains Python code for Parallelization strategies used in Machine Learning 

## Setting up python env
Run `./bin/setup_venv` for setting up virtual environment for runnning experimentations.  

## 01_Vanilla_ML.ipynb
I have written this file to recreate what i have learnt in Foundation of AI course. I am using a single GPU to run a classification network. I am using MNIST had-written numbers dataset for training and testing.

Model used is a simple feed-forward classifier network which does not employ any type of  Convolution layer. Model uses a flatten image as input passed in through a neural architecture using three hidden layers. 
   
Learning rate is kept low, as well as SGD (Stocastic Gradient Decent) is used for back propogation. These choices make sure that the Model reaches convergence very slowly(in a duration of 20 Epochs).

Training time is recorded to be around `382.6086 seconds` and output curves show smooth learning.

## 02_Model_Parallel_ML.ipynb
This file uses the same model as the one in Vanilla implementaion but divides the model computation between two GPUs. In this experiment i am using two GPUs to run the same classification network. I am still using MNIST dataset of handwritten numbers.

Model is now divided into two parts. Part-1 of the model incudes flattening of the image tensor and passing it through a feed-forward network. Part-2 of the model includes two computation layers of feed-forawrd network to output classification. 

Learning rate and Backpropogation criterion are kept the same as vanilla version for obtaining a valid comparision result. 

Training time is recorded to be around `380.9114 seconds` and output curves show smooth learning. The time take is smaller than that vanilla version but still remains significantly unchanged(difference is **-0.4%** as as compared to vanilla implementation) 

