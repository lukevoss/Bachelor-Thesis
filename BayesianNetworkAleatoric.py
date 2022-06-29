from cProfile import label
from matplotlib import style
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Conv1D, Bidirectional, LSTM, Embedding, GlobalMaxPooling1D
from tensorflow.keras import Sequential

tfd = tfp.distributions
tfpl = tfp.layers

#create Data
x_100 = np.linspace(-1, 1, 100)
y_100 = x_100 + np.random.randn(x_100.shape[0]) * 0.2
x_100 = x_100.reshape(-1,1)
y_100 = y_100.reshape(-1,1)

#create Model
event_shape = 1
model_100 = Sequential([
    Dense(32),
    Dense(units= tfpl.IndependentNormal.params_size(event_shape)),
    tfpl.IndependentNormal(event_shape)
])

#loss function
negloglik = lambda y, p_y: -p_y.log_prob(y)

model_100.compile(optimizer=tf.optimizers.Adam(learning_rate = 0.05),
              loss=negloglik)

model_100.fit(x_100, y_100, epochs=600);

model_100.summary()


# get std and mean of model
model_distribution = model_100(x_100)
model_means = model_distribution.mean()
model_std = model_distribution.stddev()

y_m2sd = model_means - 2 * model_std
y_p2sd = model_means + 2 * model_std

plt.scatter(x_100, y_100, s = 30, alpha = 1, marker = "o", color = 'red', label = 'data')
plt.plot(x_100,x_100, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'target function')
plt.plot(x_100, model_means, color= 'cornflowerblue', alpha=0.8, linewidth = 3, label='learned model $\mu$')
plt.fill_between(x_100[:,0], y_m2sd[:,0], y_p2sd[:,0], alpha = 0.4, color='skyblue', label='learned model $\mu \pm 2 \sigma$')
plt.legend()
plt.show()

