import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator



testnn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))