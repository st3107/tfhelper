import sys

print("Hi, I am '{}'.".format(sys.executable))

import tensorflow as tf

print("I found {} GPU(s) Available.".format(len(tf.config.list_physical_devices('GPU'))))
