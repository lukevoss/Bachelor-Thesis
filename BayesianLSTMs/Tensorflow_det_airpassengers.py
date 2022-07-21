import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

# from keras.models import Sequential
# from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



#Import
training_set = pd.read_csv('airline-passengers.csv')#in current working directory

training_set = training_set.iloc[:,1:2].values

#create sequenced data size(Batchsize, Sequence, feature)
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

#preprocessing data
sc = StandardScaler()
training_data = sc.fit_transform(training_set)

seq_length = 4
x, y = sliding_windows(training_data, seq_length)

trainX, testX, trainY, testY = train_test_split(x,
                                                    y,
                                                    test_size=.3,
                                                    random_state=42,
                                                    shuffle=False)

#70% Training 30% Testing
train_size = int(len(y) * 0.7)
test_size = len(y) - train_size

months = np.arange(start=0, stop=len(y),step = 1)
train_months = np.arange(start=0,stop=train_size,step=1)
test_months = np.arange(start=train_size, stop=train_size+test_size, step=1)

y_plot = sc.inverse_transform(y)

model = Sequential([
    tf.keras.layers.LSTM(2),
    Dense(1)   
])


model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(learning_rate = 0.001))
model.fit(trainX, trainY, epochs=1000, batch_size=8)

testPredict = model.predict(testX)
testPredict = sc.inverse_transform(testPredict)

plt.plot(y_plot, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'Ground Truth')
plt.scatter(months,y_plot, s = 30, alpha = 1, marker = "o", color = 'red', label = 'Data')
plt.axvline(x=train_size, c='r', linestyle='--')

#plot predictions

plt.plot(test_months,testPredict, label = 'Prediction')
plt.suptitle('Time-Series Prediction')
plt.legend()
plt.show()