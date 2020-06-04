import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []
# generating random data
for i in range(1000):
    # young people with no side effect
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # old people with side effect
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in range(50):
    # young people with side effect
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # old people without side effect
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

train_samples = np.array(train_samples)
train_labels = np.array(train_labels)

# to make net easier to learn
scaler = MinMaxScaler(feature_range=(0, 1))
# reshaping data into -1, 1
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
print(scaled_train_samples)

# this model has 2 hidden layers of 16 nodes and 32 nodes
model = Sequential([
    # dense layers are fully connected layers
    # 16 nodes, shape of data, name of activation function
    Dense(16, input_shape=(1,), activation="relu"),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax")
])

# Adam is an optimizer, loss function, what's printed out when we print out our model
model.compile(Adam(lr=.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.optimizer.lr = 0.01     # this is how to update learning rate after model is compiled
# numpy array that holds data, labels, batch size, epoch, shuffle, how much output when we see
# validation split: extract 20% from training data and use it for validation
# could also create a separate data (a list of tuples) for validation_data parameter
model.fit(scaled_train_samples, train_labels, validation_split=0.20, batch_size=10,
          epochs=20, shuffle=True, verbose=2)

test_samples = []
# Generate test samples
for i in range(105):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)

    random_older = randint(65, 100)
    test_samples.append(random_older)

test_samples = np.array(test_samples)
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))


# Predict
predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)
