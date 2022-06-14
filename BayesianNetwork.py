import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

tfd = tfp.distributions
tfpl = tfp.layers

x_100 = np.linspace(-1, 1, 100)
y_100 = x_100 + np.random.randn(x_100.shape[0]) * 0.2
#x_100 = x_100.reshape(-1,1)
#y_100 = y_100.reshape(-1,1)

#Set up Priors and Posteriors
def prior(kernel_size, bias_size, dtype = None):
    n = kernel_size + bias_size # num of params
    return Sequential([
       tfpl.DistributionLambda(
           #Normal Distribution as prior
           lambda t: tfd.Normal(loc = tf.zeros(n), scale= tf.ones(n)) #L
       )                     
  ])

def posterior(kernel_size, bias_size, dtype = None):
    n = kernel_size + bias_size # num of params
    return Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n),
    ])

event_shape = 1
model_100 = Sequential([
    tfpl.DenseVariational(input_shape = (1,),
                          units = 1,
                          make_prior_fn = prior,
                          make_posterior_fn = posterior,
                          kl_weight = 1 / x_100.shape[0]),


    Dense(units= tfpl.IndependentNormal.params_size(event_shape)),
    tfpl.IndependentNormal(event_shape)
])

#loss function negative log likelihood
nll = lambda y, p_y: -p_y.log_prob(y)

model_100.compile(optimizer=tf.optimizers.Adam(learning_rate = 0.05),
              loss=nll)
model_100.fit(x_100, y_100, epochs=200);
model_100.summary()
model_100.evaluate(x_100, y_100)

# get std and mean of model


ensemble_size = 5

plt.figure(1)
plt.scatter(x_100, y_100, s = 70, alpha = 0.3, marker = "o", label = 'Data', color = 'gray')
for _ in range(ensemble_size):
    model_distribution = model_100(x_100)
    model_means = model_distribution.mean().numpy()
    model_std = model_distribution.stddev().numpy()
    y_m2sd = model_means - 2 * model_std
    y_p2sd = model_means + 2 * model_std

    plt.plot(x_100, model_means, color='red', alpha=0.8)
    plt.plot(x_100, y_m2sd, color='green', alpha=0.8, linewidth = 2)
    plt.plot(x_100, y_p2sd, color='green', alpha=0.8, linewidth = 2)           
plt.legend()
plt.show()

###################Try if epistemic uncertainty can be reduced by more data ####################

x_1000 = np.linspace(-1, 1, 1000)
y_1000 = x_1000 + np.random.randn(x_1000.shape[0]) * 0.2

model_1000 = Sequential([
    tfpl.DenseVariational(input_shape = (1,),
                          units = 1,
                          make_prior_fn = prior,
                          make_posterior_fn = posterior,
                          kl_weight = 1 / x_1000.shape[0]),
    Dense(units= tfpl.IndependentNormal.params_size(event_shape)),
    tfpl.IndependentNormal(event_shape)
])
model_1000.compile(loss = nll, optimizer = tf.keras.optimizers.Adam(lr = 0.05))

model_1000.summary()

model_1000.fit(x_1000, y_1000, epochs=500)
model_1000.evaluate(x_1000, y_1000)

ensemble_size = 5

plt.figure(2)
plt.scatter(x_1000, y_1000, s = 70, alpha = 0.3, marker = "o", label = 'Data', color = 'gray')
for _ in range(ensemble_size):
    model_distribution = model_1000(x_1000)
    model_means = model_distribution.mean().numpy()
    model_std = model_distribution.stddev().numpy()
    y_m2sd = model_means - 2 * model_std
    y_p2sd = model_means + 2 * model_std

    plt.plot(x_1000, model_means, color='red', alpha=0.8)
    plt.plot(x_1000, y_m2sd, color='green', alpha=0.8, linewidth = 2)
    plt.plot(x_1000, y_p2sd, color='green', alpha=0.8, linewidth = 2)                
plt.legend()
plt.show()

# Plot comparison of difference in dataset size

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
for _ in range(5):
    ax1.scatter(x_100, y_100, s = 70, alpha = 0.3, marker = "o", color = 'gray')
    for _ in range(ensemble_size):
        model_distribution = model_100(x_100)
        model_means = model_distribution.mean().numpy()
        model_std = model_distribution.stddev().numpy()
        y_m2sd = model_means - 2 * model_std
        y_p2sd = model_means + 2 * model_std

        ax1.plot(x_100, model_means, color='red', alpha=0.8)
        ax1.plot(x_100, y_m2sd, color='green', alpha=0.8, linewidth = 2)
        ax1.plot(x_100, y_p2sd, color='green', alpha=0.8, linewidth = 2)
    ax1.set_title('100 Data Points')
    
    ax2.scatter(x_1000, y_1000, s = 70, alpha = 0.03, marker = "o", color = 'gray')
    for _ in range(ensemble_size):
        model_distribution = model_1000(x_1000)
        model_means = model_distribution.mean().numpy()
        model_std = model_distribution.stddev().numpy()
        y_m2sd = model_means - 2 * model_std
        y_p2sd = model_means + 2 * model_std

        ax2.plot(x_1000, model_means, color='red', alpha=0.8)
        ax2.plot(x_1000, y_m2sd, color='green', alpha=0.8, linewidth = 2)
        ax2.plot(x_1000, y_p2sd, color='green', alpha=0.8, linewidth = 2) 
    ax2.set_title('1000 Data Points')
    
    fig.suptitle('Epistemic Uncertainty Comparison')
plt.show()