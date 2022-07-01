from importlib.util import module_for_loader
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential


tfd = tfp.distributions
tfpl = tfp.layers

NUM_POINTS = 1000
x = np.linspace(-5, 2, NUM_POINTS)
y_target = 4 * x * np.cos(np.pi * np.sin(x)) + 1
y = y_target + np.random.randn(x.shape[0]) * 0.5


def prior(kernel_size, bias_size, dtype = None):
    n = kernel_size + bias_size # num of params
    return Sequential([
       tfpl.DistributionLambda(
           lambda t: tfd.Normal(loc = tf.zeros(n), scale= 2 * tf.ones(n))
       )                     
  ])

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(

          tfd.Normal(loc=t[..., :n],
                     scale= 1e-6 + 0.001 * tf.nn.softplus(t[..., n:])),
          reinterpreted_batch_ndims=1)),
    ])

event_shape = 1
model_non_linear = Sequential([
    
    tfpl.DenseVariational(input_shape = (1,),
                          units = 128,
                          make_prior_fn = prior,
                          make_posterior_fn = posterior,
                          kl_weight = 1 / x.shape[0], activation = tf.nn.silu),
    
    tfpl.DenseVariational(units = 64,
                          make_prior_fn = prior,
                          make_posterior_fn = posterior,
                          kl_weight = 1 / x.shape[0], activation = tf.nn.silu),

    tfpl.DenseVariational(units = 32,
                          make_prior_fn = prior,
                          make_posterior_fn = posterior,
                          kl_weight = 1 / x.shape[0], activation = tf.nn.silu),

    tfpl.DenseVariational(units = 16,
                          make_prior_fn = prior,
                          make_posterior_fn = posterior,
                          kl_weight = 1 / x.shape[0], activation = tf.nn.silu),

    Dense(units= tfpl.IndependentNormal.params_size(event_shape)),
    tfpl.IndependentNormal(event_shape)


])
nll = lambda y, p_y: -p_y.log_prob(y)

model_non_linear.compile(optimizer=tf.optimizers.Adam(learning_rate = 0.001),
              loss=nll)
model_non_linear.summary()
model_non_linear.fit(x, y, epochs=1000)

ensemble_size = 20

# plt.figure(0)
# plt.scatter(x, y, s = 30, alpha = 1, marker = "o", color = 'red', label = 'Data')
# plt.plot(x,y_target, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'Target function')
# for i in range(ensemble_size):
#     #define upper bounds and lower bound of epistemic uncertainty
#     if i == 0:
#         y_upper = model_non_linear(x)
#         y_lower = y_upper
#     else:
#         y_upper = np.maximum(y_upper, model_non_linear(x))  
#         y_lower = np.minimum(y_lower, model_non_linear(x))        
# plt.fill_between(x, y_upper[:,0], y_lower[:,0], alpha = 0.6, color='royalblue', label='epistemic uncertainty')      
# plt.legend()
# plt.show()

plt.figure(0)
plt.scatter(x, y, s = 30, alpha = 1, marker = "o", color = 'red', label = 'Data')
plt.plot(x,y_target, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'Target function')

y_mean_list = np.zeros((NUM_POINTS, ensemble_size))
y_upper_list = np.zeros((NUM_POINTS, ensemble_size))
y_lower_list = np.zeros((NUM_POINTS, ensemble_size))
for i in range(ensemble_size):
    model_distribution = model_non_linear(x)
    model_means = model_distribution.mean().numpy()
    #plt.plot(x,model_means, color='blue')
    y_mean_list[:,i] = model_means[:,0]
    #standard deviation
    model_std = model_distribution.stddev().numpy()
    y_std_upper = model_means + 2 * model_std
    y_std_lower = model_means - 2 * model_std
    #plt.plot(x,y_std_upper, color='green')
    y_upper_list[:,i] = y_std_upper[:,0]
    y_lower_list[:,i] = y_std_lower[:,0]
y_mean = np.mean(y_mean_list, axis=1)
y_upper = np.mean(y_upper_list, axis=1)
y_lower = np.mean(y_lower_list, axis=1)
plt.plot(x, y_mean, color= 'cornflowerblue', alpha=0.8, linewidth = 3, label='learned model $\mu$')
plt.fill_between(x, y_upper, y_lower, alpha = 0.4, color='skyblue', label='Standard Deviation')
plt.legend()
plt.show()