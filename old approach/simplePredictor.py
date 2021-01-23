import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
from keras.models import load_model

################ necessary dataframes ############
data = pd.read_csv('data_withFasterParser.csv')

dataset = data.dropna() # cleans the dataset of any incomplete datapoints


############ split data into train and test sets 80/20 ##############
train_size = int(len(dataset) * .8)

train_data = dataset[:train_size]
test_data = dataset[train_size:]

################## separate label ##################
train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop('remainingTraceTime')
test_labels = test_features.pop('remainingTraceTime')

################ load model ###############
dnn_model = keras.models.load_model('dnn_model_saved')

################# predictions on test set ##################

test_predictions = dnn_model.predict(test_features).flatten()

def plot_prediction():
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions, facecolors='none', edgecolors='r')
    plt.xlabel('True Values [Remaining time]')
    plt.ylabel('Predictions [Remaining time]')
    lims = [0, 75000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    
plot_prediction()

plt.hist(test_predictions, bins=100)
plt.xlabel('Predictions [Remaining time]')
_ = plt.ylabel('Count')
plt.show()

################ error distribution ################
error = test_predictions - test_labels

plt.hist(error, bins=100)
plt.xlabel('Prediction Error [Remaining time]')
_ = plt.ylabel('Count')
plt.show()

rmse = tf.math.sqrt(np.median(tf.square(test_predictions - test_labels)))
print(rmse)

# all_mse = np.square(test_predictions - test_labels)
# plt.hist(all_mse, bins=100)
# plt.xlabel('Prediction square error [Remaining time]')
# _ = plt.ylabel('Count')
# plt.show()