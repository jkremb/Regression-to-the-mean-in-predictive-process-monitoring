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



################### converting events to one-hot columns
dataset = data.dropna() # cleans the dataset of any incomplete datapoints
# timeAndNumOfEvents = data.drop(labels='loanAmount', axis='columns')
# timeAndNumOfEvents = timeAndNumOfEvents.drop(labels='events', axis='columns')
# timeAndNumOfEvents.dropna()   

# The columns need to be instantiated like this unfortunately.....
dataset['W_Completeren aanvraag-COMPLETE'] = ''
dataset['W_Completeren aanvraag-START'] = ''
dataset['W_Nabellen offertes-COMPLETE'] = ''
dataset['W_Nabellen offertes-START'] = ''
dataset['A_SUBMITTED-COMPLETE'] = ''
dataset['A_PARTLYSUBMITTED-COMPLETE'] = ''
dataset['W_Nabellen incomplete dossiers-COMPLETE'] = ''
dataset['W_Nabellen incomplete dossiers-START'] = ''
dataset['W_Valideren aanvraag-COMPLETE'] = ''
dataset['W_Valideren aanvraag-START'] = ''
dataset['A_DECLINED-COMPLETE'] = ''
dataset['W_Completeren aanvraag-SCHEDULE'] = ''
dataset['A_PREACCEPTED-COMPLETE'] = ''
dataset['O_SELECTED-COMPLETE'] = ''
dataset['O_CREATED-COMPLETE'] = ''
dataset['O_SENT-COMPLETE'] = ''
dataset['W_Nabellen offertes-SCHEDULE'] = ''
dataset['W_Afhandelen leads-COMPLETE'] = ''
dataset['W_Afhandelen leads-START'] = ''
dataset['A_ACCEPTED-COMPLETE'] = ''
dataset['W_Valideren aanvraag-SCHEDULE'] = ''
dataset['A_FINALIZED-COMPLETE'] = ''
dataset['W_Afhandelen leads-SCHEDULE'] = ''
dataset['O_CANCELLED-COMPLETE'] = ''
dataset['O_SENT_BACK-COMPLETE'] = ''
dataset['A_CANCELLED-COMPLETE'] = ''
dataset['W_Nabellen incomplete dossiers-SCHEDULE'] = ''
dataset['A_REGISTERED-COMPLETE'] = ''
dataset['A_APPROVED-COMPLETE'] = ''
dataset['A_ACTIVATED-COMPLETE'] = ''
dataset['O_ACCEPTED-COMPLETE'] = ''
dataset['O_DECLINED-COMPLETE'] = ''
dataset['W_Beoordelen fraude-START'] = ''
dataset['W_Beoordelen fraude-COMPLETE'] = ''
dataset['W_Beoordelen fraude-SCHEDULE'] = ''
dataset['W_Wijzigen contractgegevens-SCHEDULE'] = ''
print(dataset)


# for x in range(0,20):
#     dataset['A_SUBMITTED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_SUBMITTED'")
#     dataset['A_PARTLYSUBMITTED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_PARTLYSUBMITTED'")
#     dataset['W_Nabellen offertes-START'][x] = origin[x].count("'START', 'W_Nabellen offertes'")
#     dataset['W_Afhandelen leads-SCHEDULE'][x] = origin[x].count("'SCHEDULE', 'W_Afhandelen leads'")

# by origin i mean events
origin = dataset.pop('events')

for x in range(0,len(dataset)):
    dataset['W_Completeren aanvraag-COMPLETE'][x] = origin[x].count("'COMPLETE', 'W_Completeren aanvraag'")
    dataset['W_Completeren aanvraag-START'][x] = origin[x].count("'START', 'W_Completeren aanvraag'")
    dataset['W_Nabellen offertes-COMPLETE'][x] = origin[x].count("'COMPLETE', 'W_Nabellen offertes'")
    dataset['W_Nabellen offertes-START'][x] = origin[x].count("'START', 'W_Nabellen offertes'")
    dataset['A_SUBMITTED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_SUBMITTED'")
    dataset['A_PARTLYSUBMITTED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_PARTLYSUBMITTED'")
    dataset['W_Nabellen incomplete dossiers-COMPLETE'][x] = origin[x].count("'COMPLETE', 'W_Nabellen incomplete dossiers'")
    dataset['W_Nabellen incomplete dossiers-START'][x] = origin[x].count("'START', 'W_Nabellen incomplete dossiers'")
    dataset['W_Valideren aanvraag-COMPLETE'][x] = origin[x].count("'COMPLETE', 'W_Valideren aanvraag'")
    dataset['W_Valideren aanvraag-START'][x] = origin[x].count("'START', 'W_Valideren aanvraag'")
    dataset['A_DECLINED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_DECLINED'")
    dataset['W_Completeren aanvraag-SCHEDULE'][x] = origin[x].count("'SCHEDULE', 'W_Completeren aanvraag'")
    dataset['A_PREACCEPTED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_PREACCEPTED'")
    dataset['O_SELECTED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'O_SELECTED'")
    dataset['O_CREATED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'O_CREATED'")
    dataset['O_SENT-COMPLETE'][x] = origin[x].count("'COMPLETE', 'O_SENT'")
    dataset['W_Nabellen offertes-SCHEDULE'][x] = origin[x].count("'SCHEDULE', 'W_Nabellen offertes'")
    dataset['W_Afhandelen leads-COMPLETE'][x] = origin[x].count("'COMPLETE', 'W_Afhandelen leads'")
    dataset['W_Afhandelen leads-START'][x] = origin[x].count("'START', 'W_Afhandelen leads'")
    dataset['A_ACCEPTED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_ACCEPTED'")
    dataset['W_Valideren aanvraag-SCHEDULE'][x] = origin[x].count("'SCHEDULE', 'W_Valideren aanvraag'")
    dataset['A_FINALIZED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_FINALIZED'")
    dataset['W_Afhandelen leads-SCHEDULE'][x] = origin[x].count("'SCHEDULE', 'W_Afhandelen leads'")
    dataset['O_CANCELLED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'O_CANCELLED'")
    dataset['O_SENT_BACK-COMPLETE'][x] = origin[x].count("'COMPLETE', 'O_SENT_BACK'")
    dataset['A_CANCELLED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_CANCELLED'")
    dataset['W_Nabellen incomplete dossiers-SCHEDULE'][x] = origin[x].count("'SCHEDULE', 'W_Nabellen incomplete dossiers'")
    dataset['A_REGISTERED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_REGISTERED'")
    dataset['A_APPROVED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_APPROVED'")
    dataset['A_ACTIVATED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'A_ACTIVATED'")
    dataset['O_ACCEPTED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'O_ACCEPTED'")
    dataset['O_DECLINED-COMPLETE'][x] = origin[x].count("'COMPLETE', 'O_DECLINED'")
    dataset['W_Beoordelen fraude-START'][x] = origin[x].count("'START', 'W_Beoordelen fraude'")
    dataset['W_Beoordelen fraude-COMPLETE'][x] = origin[x].count("'COMPLETE', 'W_Beoordelen fraude'")
    dataset['W_Beoordelen fraude-SCHEDULE'][x] = origin[x].count("'SCHEDULE', 'W_Beoordelen fraude'")
    dataset['W_Wijzigen contractgegevens-SCHEDULE'][x] = origin[x].count("'SCHEDULE', 'W_Wijzigen contractgegevens'")

print(dataset)
dataset.to_csv('processedData.csv')

# print(dataset.tail(6))
# timeAndNumOfEvents.plot.scatter(x='numberOfEvents', y='remainingTraceTime')
# plt.show();