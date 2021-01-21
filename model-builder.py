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

# pd.set_option('display.max_columns', None)
# print(dataset.head(5))
dataset = data.dropna() # cleans the dataset of any incomplete datapoints
dataset.multiply(1.0)

############ split data into train and test sets 80/20 ##############
train_size = int(len(dataset) * .8)

train_data = dataset[:train_size]
test_data = dataset[train_size:]
# print(test_data)

################## separate label ##################
train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop('remainingTraceTime')
test_labels = test_features.pop('remainingTraceTime')

################# normalize data ########################
# train_stats = train_data.describe()
# train_stats = train_stats.transpose()
# print(train_stats)

# def norm(x):
#     return (x - train_stats['mean']) / train_stats['std']   #### normalize with median? how?
# normed_train_data = norm(train_data)
# normed_test_data = norm(test_data)
# print(normed_train_data)

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# print(normalizer.mean.numpy())

# first = np.array(train_features[:1])
# with np.printoptions(precision=2, suppress=True):
#   print('First example:', first)
#   print()
#   print('Normalized:', normalizer(first).numpy())

################# build model (linear) ####################
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


############### train model #############

history = linear_model.fit(
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
# def plot_loss(history):
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.ylim([8000, 14000])
#     plt.xlabel('Epoch')
#     plt.ylabel('Error [Trace Time Remaining]')
#     plt.legend()
#     plt.grid(True)

# plot_loss(history)
# plt.show()

################# predictions on test set ##################

test_predictions = linear_model.predict(test_features).flatten()

def plot_prediction():
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [Remaining time]')
    plt.ylabel('Predictions [Remaining time]')
    lims = [0, 75000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    # plt.show()
    
# plot_prediction()


################ error distribution ################
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [Remaining time]')
_ = plt.ylabel('Count')
plt.show()





# ############## save the model

# history.save('model')


# test_results = {}
# test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose=1)
# print(test_results)

'''
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
        # layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])

    return model

# reinsert this later
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model = build_model()

print(model.summary())

########## testing with batch of training data ###############
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)



################# train model ##################
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 200

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, callbacks=[early_stop, PrintDot()]
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail(10))

########### plotting learning curves for tweaking ###########
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.legend()
    plt.ylim([12000,16000])

plot_history(history)
plt.show()

############ Finding mean abs error #################
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} minutes remaining".format(mae))

############# Predicting time remaining values using testing set ###############
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [trace time remaining]')
plt.ylabel('Predicted Values [trace time remaining]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

########## error testing (Gaussian normal dist would be perfect) ############
# error = test_predictions - test_labels
# plt.hist(error, bin = 25)
# plt.xlabel('Prediction Error [trace time remaining]')
# _ = plt.ylabel('Count')
# plt.show()
'''