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
print('Cutoff date is: ')
print(cutoff_date)
print(type(cutoff_date))

# print(train_data)
# # print(train_data["loanAmount"])
# print((train_data.firstTraceEventTimestamp < cutoff_date))
# print(train_data[train_data.firstTraceEventTimestamp < cutoff_date])
# print((train_data[train_data.firstTraceEventTimestamp < cutoff_date]).index)
# 2011-10-01 09:45:37.274
train_data = train_data.drop((train_data[train_data.lastTraceEventTimestamp >= cutoff_date]).index)
print(train_data)


# print(train_data)

################## separate label ##################

################# normalize data ########################


################## build model (Deep neural network) #############


############### train model #############


##################### plot the loss curves ##############



################# predictions on test set ##################




