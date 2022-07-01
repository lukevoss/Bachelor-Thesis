import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Normal

def compute_KL_univariate_prior(univariateprior, theta, sample):

        """
        :param prior:  assuming univariate prior of Normal(m,s);
        :param posterior: (theta: mean,std) to create posterior q(w/theta) i.e. Normal(mean,std)
        :param sample: Number of sample
        """
        sample=tf.reshape(sample, [-1])  #flatten vector
        (mean,std)=theta
        mean =tf.reshape(mean, [-1])
        std=tf.reshape(std, [-1])
        posterior = Normal(mean, std)

        (mean2,std2) = univariateprior
        prior=Normal(mean2, std2)

        q_theta=tf.reduce_sum(posterior.log_prob(sample))
        p_d=tf.reduce_sum(prior.log_prob(sample))

        KL=tf.subtract(q_theta,p_d)

        return KL



def variationalPosterior(shape, name, prior, istraining):

    mu=tf.get_variable("{}_mean".format(name), shape=shape, dtype=tf.float32);
    rho=tf.get_variable("{}_rho".format(name), shape=shape, dtype=tf.float32);
    sigma = tf.math.softplus(rho)

    #if training we add noise to variation parameters theta
    if (istraining):
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