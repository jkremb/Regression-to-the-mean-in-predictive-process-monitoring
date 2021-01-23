import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
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

############## drop trace at cutoff date from test set
cutoff_date = test_data.iloc[0,2]
test_data = test_data.drop((test_data[test_data.firstTraceEventTimestamp == cutoff_date]).index)


def predict_for_prefix_length_n(n):

    # test model prefix_n only for test datapoints with prefix n
    # also omit test datapoints with whos tracelength is n, n+1 and n+2, as the prediction are too easy
    test_features = test_data.drop((test_data[test_data.currentPrefixLength != n]).index)
    test_features = test_features.drop((test_features[test_features.totalPrefixLength == n]).index)
    test_features = test_features.drop((test_features[test_features.totalPrefixLength == (n+1)]).index)
    test_features = test_features.drop((test_features[test_features.totalPrefixLength == (n+2)]).index)

    ############ drop redundant columns ####################
    test_features_n = test_features.drop(['firstTraceEventTimestamp','lastTraceEventTimestamp','totalPrefixLength','currentPrefixLength'],
                                                axis='columns')

    ################## separate labels ##################
    test_features_copy = test_features_n.copy()
    test_labels = test_features_copy.pop('remainingTraceTime')

    ################ load model ###############
    dnn_model = keras.models.load_model('dnn_model_saved_prefix_length_' + str(n))

    ################# run predictions on test set ##################
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

    # plt.hist(test_predictions, bins=100)
    # plt.xlabel('Predictions [Remaining time]')
    # _ = plt.ylabel('Count')
    # plt.show()

    # ################ error distribution ################
    # error = test_predictions - test_labels

    # plt.hist(error, bins=100)
    # plt.xlabel('Prediction Error [Remaining time]')
    # _ = plt.ylabel('Count')
    # plt.show()

    ##########--------------------- Regression to the mean adjustment ---------------------##########
    confidence = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for c in confidence:
        mse = DataFrame.mean(np.square(test_predictions - test_labels))
        print('MSE for confidence '+ str(c) + ' is ' + str(mse))


        adjusted_predictions = c*(test_predictions) + DataFrame.median(test_labels)*(1-c)
        # adjusted_predictions = confidence[0]*(test_predictions) + DataFrame.median(test_labels)*(1 - confidence[0])
        # median of predic

        mse_adjusted = DataFrame.mean(np.square(adjusted_predictions - test_labels))
        print('Adjusted MSE for confidence '+ str(c) + ' is ' + str(mse_adjusted))


################## predict all models with separate prefixes ########
# prefix_lengths = [3, 6, 8, 10, 12, 14, 16, 18, 20, 30, 50]
# for prefix in prefix_lengths:
#     predict_for_prefix_length_n(prefix)
predict_for_prefix_length_n(12)


#################
# confidence = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# mse = DataFrame.mean(np.square(test_predictions - test_labels))

# adjusted_prediction = c*(test_predictions) + DataFrame.median(test_labels)
# # median of predic

# mse_adjusted = DataFrame.mean(np.square(test_predictions - test_labels))