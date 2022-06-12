import numpy as np
import tensorflow as tf
#from tensorflow.keras.layers import LSTMCell
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Normal

def variationalPosterior(shape, name, prior, is_training):
    """
    this function create a variational posterior q(w/theta) over a given "weight:w" of the network
    theta is parameterized by mean+standard*noise we apply the reparameterization trick from kingma et al, 2014
    with correct loss function (free energy) we learn mean and standard to estimate of theta, thus can estimate
    posterior p(w/D) by computing KL loss for each variational posterior q(w/theta) with prior(w)
    :param name: is the name of the tensor/variable to create variational posterior  q(w/Q) for true posterior (p(w/D))
    :param shape: is the shape of the weight variable
    :param training: whether in training or inference mode
    :return: samples (i.e. weights), mean of weights, std in-case of the training there is noise add to the weights
    """
    # theta=mu+sigma i.e. theta = mu+sigma i.e. mu+log(1+exp(rho)), log(1+exp(rho))
    # is the computed by using tf.math.softplus(rho)


    mu=tf.get_variable("{}_mean".format(name), shape=shape, dtype=tf.float32);
    rho=tf.get_variable("{}_rho".format(name), shape=shape, dtype=tf.float32);
    sigma = tf.math.softplus(rho)

    #if training we add noise to variation parameters theta
    if (is_training):
        epsilon= Normal(0,1.0).sample(shape)
        sample=mu+sigma*epsilon
    else:
        sample=mu+sigma;

    theta=(mu,sigma)

    kl_loss = compute_KL_univariate_prior(prior, theta, sample)

    tf.summary.histogram(name + '_rho_hist', rho)
    tf.summary.histogram(name + '_mu_hist', mu)
    tf.summary.histogram(name + '_sigma_hist', sigma)

    # we shall used this in the training to get kl loss
    tf.add_to_collection("KL_layers", kl_loss)

    return sample, mu, sigma


class BayesianLSTMCell(tf.keras.layers.LSTMCell):

    def __init__(self, num_units, prior, is_training, name, **kwargs):

        super(BayesianLSTMCell, self).__init__(num_units, **kwargs)

        self.w = None
        self.b = None
        self.prior = prior
        self.layer_name = name
        self.isTraining = is_training
        self.num_units = num_units
        self.kl_loss=None

        print("Creating lstm layer:" + name)


    def call(self, inputs, state):

        if self.w is None:

            size = inputs.get_shape()[-1].value
            self.w, self.w_mean, self.w_sd = variationalPosterior((size+self.num_units, 4*self.num_units), self.layer_name+'_weights', self.prior, self.isTraining)
            self.b, self.b_mean, self.b_sd = variationalPosterior((4*self.num_units,1), self.layer_name+'_bias', self.prior, self.isTraining)

        cell, hidden = state
        concat_inputs_hidden = tf.concat([inputs, hidden], 1)
        concat_inputs_hidden = tf.nn.bias_add(tf.matmul(concat_inputs_hidden, self.w), tf.squeeze(self.b))
        # Gates: Input, New, Forget and Output
        i, j, f, o = tf.split(value=concat_inputs_hidden, num_or_size_splits=4, axis=1)
        new_cell = (cell * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
        new_hidden = self._activation(new_cell) * tf.sigmoid(o)
        new_state = LSTMStateTuple(new_cell, new_hidden)

        return new_hidden, new_state
