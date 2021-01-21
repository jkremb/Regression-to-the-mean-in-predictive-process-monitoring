import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data = pd.read_csv('data_withFasterParser.csv')

dataset = data.dropna() # cleans the dataset of any incomplete datapoints
dataset.multiply(1.0)

############ split data into train and test sets 80/20 ##############
train_size = int(len(dataset) * .8)

train_data = dataset[:train_size]
test_data = dataset[train_size:]

################## separate label ##################
train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop('remainingTraceTime')
test_labels = test_features.pop('remainingTraceTime')

################# normalize data ########################
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# print(normalizer.mean.numpy())

# first = np.array(train_features[:1])
# with np.printoptions(precision=2, suppress=True):
#   print('First example:', first)
#   print()
#   print('Normalized:', normalizer(first).numpy())

################## build model (Deep neural network) #############
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

############### train model #############

history = dnn_model.fit(
    train_features, train_labels, 
    epochs=100,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

##################### plot the loss curves ##############
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([8000, 14000])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Trace Time Remaining]')
    plt.legend()
    plt.grid(True)

plot_loss(history)
plt.show()

################# predictions on test set ##################

test_predictions = dnn_model.predict(test_features).flatten()

def plot_prediction():
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [Remaining time]')
    plt.ylabel('Predictions [Remaining time]')
    lims = [0, 75000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    
plot_prediction()


################ error distribution ################
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [Remaining time]')
_ = plt.ylabel('Count')
plt.show()

