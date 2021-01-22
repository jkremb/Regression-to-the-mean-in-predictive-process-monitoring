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

################ load model ###############
dnn_model = keras.models.load_model('dnn_model_saved')
print('FLag 1 ')

################# predictions on test set ##################

# try out on single prediction point
prediction_point = pd.read_csv('testingpoint.csv')
print('Flag 2')
prediction_point_labels = prediction_point.pop('remainingTraceTime')
print('FLag 3')
print(prediction_point)

# test_predictions = dnn_model.predict(test_features).flatten()
test_predictions = dnn_model.predict(prediction_point).flatten()
print(test_predictions)


# def plot_prediction():
#     a = plt.axes(aspect='equal')
#     plt.scatter(prediction_point_labels, test_predictions)
#     # plt.scatter(test_labels, test_predictions)
#     plt.xlabel('True Values [Remaining time]')
#     plt.ylabel('Predictions [Remaining time]')
#     lims = [0, 75000]
#     plt.xlim(lims)
#     plt.ylim(lims)
#     _ = plt.plot(lims, lims)
#     plt.show()
    
# plot_prediction()


################ error distribution ################
# error = test_predictions - test_labels
# error = test_predictions - prediction_point_labels
# plt.hist(error, bins=100)
# plt.xlabel('Prediction Error [Remaining time]')
# _ = plt.ylabel('Count')
# plt.show()

