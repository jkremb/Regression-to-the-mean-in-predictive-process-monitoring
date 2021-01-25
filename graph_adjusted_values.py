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

graph = pd.DataFrame(index = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
prefix_lengths = [3, 6, 10, 12, 18, 20, 30, 50]
for n in prefix_lengths:
    data = pd.read_csv(('results/adjusted_MSE_n'+str(n)+'.csv'))
    # data = data.pop('MSE_improvement')
    graph['MSE_improvement_prefix_'+str(n)] = data['MSE_improvement'].values
    print (data)


graph.plot.line(linewidth=4)
leg = plt.legend(fontsize=20)
for line in leg.get_lines():
    line.set_linewidth(4.0)
plt.xlabel('Confidence', fontsize=18)
plt.ylabel('MSE improvement x 10^7', fontsize=18)
plt.show();