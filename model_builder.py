from datetime import datetime
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
# print(dataset['firstTraceEventTimestamp'])

############ sort traces chronologically by firstTraceEventTimestamp#######################
dataset['firstTraceEventTimestamp']= pd.to_datetime(dataset['firstTraceEventTimestamp'], format="%Y-%m-%dT%H:%M:%S.%f")
dataset['lastTraceEventTimestamp']= pd.to_datetime(dataset['lastTraceEventTimestamp'], format="%Y-%m-%dT%H:%M:%S.%f")

dataset.sort_values(by='firstTraceEventTimestamp')
# print(dataset)

dataset.to_csv('data_sortedChrono.csv')

############ split data into train and test sets 80/20 ##############
train_size = int(len(dataset) * .8)

train_data = dataset[:train_size]
test_data = dataset[train_size:]

############ remove any data from training set, which has event timestamps   #############
############ older than the first test trace to prevent clairvoyance         #############
# consider that a trace is most likely split in half
cutoff_date = test_data.iloc[0,2]
# cutoff_date = pd.to_datetime(cutoff_date, format="%Y-%m-%d %H:%M:%S.%f")
# print('Cutoff date is: ')
# print(cutoff_date)
# print(type(cutoff_date))

# print(train_data)
# # print(train_data["loanAmount"])
# print((train_data.firstTraceEventTimestamp < cutoff_date))
# print(train_data[train_data.firstTraceEventTimestamp < cutoff_date])
# print((train_data[train_data.firstTraceEventTimestamp < cutoff_date]).index)
# 2011-10-01 09:45:37.274
train_data = train_data.drop((train_data[train_data.lastTraceEventTimestamp >= cutoff_date]).index)
# print(train_data)

################## separate label ##################
train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop('remainingTraceTime')
test_labels = test_features.pop('remainingTraceTime')


def train_model_with_prefix_length_n(n):

    ############ drop datapoints where the currentprefixlength != n ##############
    train_features_n = train_features.copy()
    train_features_n = train_data.drop((train_data[train_data.currentPrefixLength != n]).index)

    ############ drop redundant columns ####################
    train_features_n = train_features_n.drop(['firstTraceEventTimestamp','lastTraceEventTimestamp','totalPrefixLength','currentPrefixLength'],
                                             axis='columns')
    # ,'lastTraceEventTimestamp','totalPrefixLength','currentPrefixLength')
    print('------------------------------------------------')
    print('Dataset with prefix length ' + str(n) + ' has ' + str(len(train_features_n)) + " datapoints")
    print('------------------------------------------------')

    ################# normalize data ########################
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features_n))

    ################## build model (Deep neural network) #############
    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',metrics=['accuracy'],
                    optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    dnn_model = build_and_compile_model(normalizer)

    ############### train model #############

    history = dnn_model.fit(
        train_features_n, train_labels, 
        epochs=100,
        # suppress logging
        verbose=1,
        # Calculate validation results on 20% of the training data
        validation_split = 0.1)

    ###################### save the model for later use ###########
    dnn_model.save('dnn_model_saved_prefix_length_'+str(n))

################## train all models with separate prefixes ########
prefix_lengths = [3, 6, 8, 10, 12, 14, 16, 18, 20, 30, 50]
for prefix in prefix_lengths:
    train_model_with_prefix_length_n(prefix)

# ##################### plot the loss curves ##############



# ################# predictions on test set ##################




