import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing



data = pd.read_csv('data/BPI_Challenge_2012.csv')

dataset = data.dropna() # cleans the dataset of any incomplete datapoints
print(dataset)

############ sort traces chronologically #######################

############ split data into train and test sets 80/20 ##############

############ remove any data from training set, which has event timestamps   #############
############ older than the first test trace to prevent clairvoyance         #############

################## separate label ##################

################# normalize data ########################


################## build model (Deep neural network) #############


############### train model #############


##################### plot the loss curves ##############



################# predictions on test set ##################




