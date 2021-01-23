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
dataset = pd.read_csv('data_sortedChrono.csv')
dataset = dataset.drop('Unnamed: 0', axis=1)

# ############ split data into train and test sets 80/20 ##############
train_size = int(len(dataset) * .8)

test_data = dataset[train_size:]

cutoff_date = test_data.iloc[0,2]

test_data = test_data.drop((test_data[test_data.firstTraceEventTimestamp == cutoff_date]).index)

################## separate label ##################

test_labels = test_data.copy().pop('remainingTraceTime')

############ drop redundant columns ####################
test_features_n = test_data.drop(['firstTraceEventTimestamp','lastTraceEventTimestamp','totalPrefixLength','currentPrefixLength'],
                                            axis='columns')


def predict_for_prefix_length_n(n):
    ################ load model ###############
    dnn_model = keras.models.load_model('dnn_model_saved_prefix_length_' + str(n))

    ################# predictions on test set ##################

    test_predictions = dnn_model.predict(test_features_n).flatten()

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

################## predict all models with separate prefixes ########
prefix_lengths = [3, 6, 8, 10, 12, 14, 16, 18, 20, 30, 50]
for prefix in prefix_lengths:
    predict_for_prefix_length_n(prefix)