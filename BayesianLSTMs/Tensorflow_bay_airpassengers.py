import numpy as np  # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import tensorflow as tf
import edward2 as ed

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import
# in current working directory
training_set = pd.read_csv('airline-passengers.csv')

training_set = training_set.iloc[:, 1:2].values

# create sequenced data size(Batchsize, Sequence, feature)


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


# preprocessing data
sc = StandardScaler()
training_data = sc.fit_transform(training_set)
#training_data = training_set

seq_length = 4
x, y = sliding_windows(training_data, seq_length)

trainX, testX, trainY, testY = train_test_split(x,
                                                y,
                                                test_size=.3,
                                                random_state=42,
                                                shuffle=False)

# 70% Training 30% Testing
train_size = int(len(y) * 0.7)
test_size = len(y) - train_size

months = np.arange(start=0, stop=len(y), step=1)
train_months = np.arange(start=0, stop=train_size, step=1)
test_months = np.arange(start=train_size, stop=train_size+test_size, step=1)

dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
dataset = dataset.batch(139)  # batch size


y_plot = sc.inverse_transform(y)
#y_plot = y

#learning rate decay
lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate= 0.1, decay_steps=100, decay_rate=0.90)

lstm_dim = 2
cell = ed.layers.LSTMCellReparameterization(lstm_dim)
model = tf.keras.Sequential([
    tf.keras.layers.RNN(cell),
    tf.keras.layers.Dense(1)])
optimizer = tf.keras.optimizers.Adam(lr_schedule)

epochs = 2000
sample_nbr = 3
criterion = tf.keras.losses.MeanSquaredError()
complexity_cost_weight = 1/x.shape[0]

for epoch in range(epochs):
    for i, (datapoints, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            for _ in range(sample_nbr):
                outputs = model(datapoints)
                loss_train = criterion(labels, outputs)
                loss_train += sum(model.losses) * complexity_cost_weight  # +kl divergence
            loss_train = loss_train / sample_nbr
        grads = tape.gradient(loss_train, model.variables)
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradients(grads_and_vars)
    if epoch % 100 == 0:
        outputs_test = model(testX)
        loss_test = criterion(testY, outputs_test)
        loss_test += sum(model.losses) * complexity_cost_weight
        print("Iteration: {} Val-loss: {:.4f} Train-loss: {:.4f}".format(str(epoch),
              loss_test, loss_train))



plt.plot(y_plot, linestyle='dashed', color='black',
         linewidth=3, label='Ground Truth')
plt.scatter(months, y_plot, s=30, alpha=1,
            marker="o", color='red', label='Data')
plt.axvline(x=train_size, c='r', linestyle='--')

# plot predictions
ensemble_size = 20

test_predict = [model(testX) for i in range(ensemble_size)]
for i in range(ensemble_size):
    prediction = test_predict[i]
    prediction = prediction.numpy()
    prediction = sc.inverse_transform(prediction)
    test_predict[i] = prediction
test_predict_mean = np.stack(test_predict)
means = test_predict_mean.mean(axis=0)
stds = test_predict_mean.std(axis=0)
y_upper = means + (2 * stds)
y_lower = means - (2 * stds)

plt.fill_between(test_months, y_upper[:, 0], y_lower[:, 0],
                 alpha=0.4, color='skyblue', label='Standard Deviation')
plt.plot(test_months, means, label='Prediction')
plt.suptitle('Time-Series Prediction')
plt.legend()
plt.show()
