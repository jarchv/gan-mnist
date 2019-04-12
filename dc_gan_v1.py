import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

dataset = input_data.read_data_sets("MNIST_data/", one_hot = True)

