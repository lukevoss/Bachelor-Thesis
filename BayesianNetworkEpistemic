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


model_100 = Sequential([
    tfpl.DenseVariational(input_shape = (1,),
                          units = 1,
                          make_prior_fn = prior,
                          make_posterior_fn = posterior,
                          kl_weight = 1 / x_100.shape[0]),
])



model_100.compile(loss = 'mse', optimizer= 'adam')
model_100.summary()
model_100.fit(x_100, y_100, epochs=1000)
model_100.evaluate(x_100, y_100)

ensemble_size = 5

plt.figure(0)
plt.scatter(x_100, y_100, s = 30, alpha = 1, marker = "o", color = 'red', label = 'Data')
plt.plot(x_100,x_100, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'Target function')
list_models = []
for i in range(ensemble_size):
    list_models.append(model_100(x_100)) 
    #define upper bounds and lower bound of epistemic uncertainty
    if i == 0:
        y_upper = model_100(x_100)
        y_lower = y_upper
    else:
        y_upper = np.maximum(y_upper, model_100(x_100))  
        y_lower = np.minimum(y_lower, model_100(x_100))        
plt.fill_between(x_100, y_upper[:,0], y_lower[:,0], alpha = 0.7, color='cornflowerblue', label='epistemic uncertainty')
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
                          kl_weight = 1 / x_1000.shape[0])
])
model_1000.compile(loss = 'mse', optimizer= 'adam')
"""
we haven't put a distribution on the output, all the uncertainties in the weights.
So we can still use a normal deterministic loss-function
"""

model_1000.summary()

model_1000.fit(x_1000, y_1000, epochs=1000)
model_1000.evaluate(x_1000, y_1000)

ensemble_size = 5

plt.figure(1)
plt.scatter(x_1000, y_1000, s = 30, alpha = 1, marker = "o", color = 'red', label = 'data')
plt.plot(x_1000,x_1000, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'target function')
list_models = []
for i in range(ensemble_size):
    list_models.append(model_1000(x_1000)) 
    #define upper bounds and lower bound of epistemic uncertainty
    if i == 0:
        y_upper = model_1000(x_1000)
        y_lower = y_upper
    else:
        y_upper = np.maximum(y_upper, model_1000(x_1000))  
        y_lower = np.minimum(y_lower, model_1000(x_1000))        
plt.fill_between(x_1000, y_upper[:,0], y_lower[:,0], alpha = 0.7, color='cornflowerblue', label='epistemic uncertainty')      
plt.legend()
plt.show()

# Plot comparison of difference in dataset size

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.scatter(x_100, y_100, s = 30, alpha = 1, marker = "o", color = 'red', label = 'Data')
ax1.plot(x_100,x_100, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'Target function')
ax2.scatter(x_1000, y_1000, s = 30, alpha = 1, marker = "o", color = 'red', label = 'data')
ax2.plot(x_1000,x_1000, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'target function')
for i in range(5):
    if i == 0:
        y_upper_100 = model_100(x_100)
        y_lower_100 = y_upper_100
        y_upper_1000 = model_1000(x_1000)
        y_lower_1000 = y_upper_1000
    else:
        y_upper_100 = np.maximum(y_upper_100, model_100(x_100))  
        y_lower_100 = np.minimum(y_lower_100, model_100(x_100)) 
        y_upper_1000 = np.maximum(y_upper_1000, model_1000(x_1000))  
        y_lower_1000 = np.minimum(y_lower_1000, model_1000(x_1000))
ax1.fill_between(x_100, y_upper_100[:,0], y_lower_100[:,0], alpha = 0.7, color='cornflowerblue', label='epistemic uncertainty')      
ax2.fill_between(x_1000, y_upper_1000[:,0], y_lower_1000[:,0], alpha = 0.7, color='cornflowerblue', label='epistemic uncertainty')         
ax1.set_title('100 Data Points')
ax2.set_title('1000 Data Points')
fig.suptitle('Epistemic Uncertainty Comparison')
plt.legend()
plt.show()