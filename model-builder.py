import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data = pd.read_csv('data.csv')
# data = shuffle(data, random_state=22)
# print(data.head(6))

## encode data
## format to matrix where 0 if certain event is not there, 1 if certain event is there
## use sklearn.MultiLabelBinarizer() for this



## simplifying code for sake of learning
timeAndNumOfEvents = data.drop(labels='loanAmount', axis='columns')
timeAndNumOfEvents = timeAndNumOfEvents.drop(labels='events', axis='columns')
timeAndNumOfEvents.dropna()   # cleans the dataset of any incomplete datapoints
print(timeAndNumOfEvents.head(6))

# timeAndNumOfEvents.plot.scatter(x='numberOfEvents', y='remainingTraceTime')
# plt.show();

############ split data into train and test sets 80/20 ##############
train_size = int(len(data) * .8)

train_data = timeAndNumOfEvents[:train_size]
test_data = timeAndNumOfEvents[train_size:]
# print(test_data)


# sns.pairplot(train_data[["traceTimeInMin", "numberOfEvents"]], diag_kind="kde")
# plt.show()

################## determining label ##################
train_labels = train_data.pop('remainingTraceTime')
test_labels = test_data.pop('remainingTraceTime')

################# normalize data ########################
train_stats = train_data.describe()
# train_stats.pop('traceTimeInMin')
train_stats = train_stats.transpose()


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_data)
normed_test_data = norm(test_data)


################# build model ####################

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])

    return model

model = build_model()

print(model.summary())

########## testing with batch of training data ###############
# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# print(example_result)



################# train model ##################
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 10

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, callbacks=[PrintDot()]
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
    plt.ylim([0,7000])

plot_history(history)
plt.show()