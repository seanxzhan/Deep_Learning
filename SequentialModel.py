from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

model = Sequential([
    # dense layers are fully connected layers
    # 5 nodes, shape of data,  activation function
    Dense(5, input_shape=(3,), activation="relu"),
    Dense(2, activation="softmax")
])
