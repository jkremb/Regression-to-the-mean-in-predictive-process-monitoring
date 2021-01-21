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

data = pd.read_csv('data_withFasterParser.csv')

dataset = data.dropna() # cleans the dataset of any incomplete datapoints
dataset.multiply(1.0)

############ split data into train and test sets 80/20 ##############
train_size = int(len(dataset) * .8)

train_data = dataset[:train_size]
test_data = dataset[train_size:]
# print(train_data[:10])

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

    model.compile(loss='mean_absolute_error',metrics=['accuracy'],
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

############### train model #############

history = dnn_model.fit(
    train_features, train_labels, 
    epochs=20,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist)

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

###################### plot the accuracy curve #############
def acc_loss(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

acc_loss(history)
plt.show()

################# predictions on test set ##################

# try out on single prediction point
prediction_point = pd.read_csv('testingpoint.csv')
prediction_point_labels = prediction_point.pop('remainingTraceTime')
print(prediction_point)

# test_predictions = dnn_model.predict(test_features).flatten()
predictions_forOnePoint = []
for x in range(1,1000):

    # test_predictions = dnn_model.predict(prediction_point).flatten()
print(test_predictions)

def plot_prediction():
    a = plt.axes(aspect='equal')
    plt.scatter(prediction_point_labels, test_predictions)
    # plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [Remaining time]')
    plt.ylabel('Predictions [Remaining time]')
    lims = [0, 75000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    
plot_prediction()


################ error distribution ################
# error = test_predictions - test_labels
error = test_predictions - prediction_point_labels
plt.hist(error, bins=100)
plt.xlabel('Prediction Error [Remaining time]')
_ = plt.ylabel('Count')
plt.show()


################ Regression to the mean postprocessing ################

# prediction from model * confidence from model + (median/all remaining times)*(1-CI)
# = prediction with rttm accounted for

# confidence intervals

# def create_dropout_predict_function(model, dropout):
#     """
#     Create a keras function to predict with dropout
#     model : keras model
#     dropout : fraction dropout to apply to all layers
    
#     Returns
#     predict_with_dropout : keras function for predicting with dropout
#     """
    
#     # Load the config of the original model
#     conf = model.get_config()
#     # Add the specified dropout to all layers
#     for layer in conf['layers']:
#         # Dropout layers
#         if layer["class_name"]=="Dropout":
#             layer["config"]["rate"] = dropout
#         # Recurrent layers with dropout
#         elif "dropout" in layer["config"].keys():
#             layer["config"]["dropout"] = dropout

#     # Create a new model with specified dropout
#     if type(model)==Sequential:
#         # Sequential
#         model_dropout = Sequential.from_config(conf)
#     else:
#         # Functional
#         model_dropout = Model.from_config(conf)
#     model_dropout.set_weights(model.get_weights()) 
    
#     # Create a function to predict with the dropout on
#     predict_with_dropout = K.function(model_dropout.inputs+[K.learning_phase()], model_dropout.outputs)
    
#     return predict_with_dropout



# dropout = 0.5
# num_iter = 20
# num_samples = input_data[0].shape[0]

# path_to_model = "../models/pretrainedmodel.hdf5"
# model = load_model(path_to_model)

# predict_with_dropout = create_dropout_predict_function(model, dropout)

# predictions = np.zeros((num_samples, num_iter))
# for i in range(num_iter):
#     predictions[:,i] = predict_with_dropout(input_data+[1])[0].reshape(-1)
