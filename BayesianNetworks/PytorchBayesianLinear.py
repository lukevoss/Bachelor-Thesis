import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


NUM_POINTS = 400
x = torch.linspace(-5, 2, NUM_POINTS)
y_target = 4 * x * np.cos(np.pi * np.sin(x)) + 1
y = y_target + torch.randn(x.shape[0]) * 0.5
input_size = 1
output_size = 1

@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 128)
        self.blinear2 = BayesianLinear(128, 64)
        self.blinear3 = BayesianLinear(64, output_dim)
        
    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.silu(x_)
        x_ = self.blinear2(x_)
        x_ = F.silu(x_)
        return self.blinear3(x_)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bayes_network = BayesianRegressor(input_size, output_size).to(device)
optimizer = optim.Adam(bayes_network.parameters(), lr=0.001)
criterion = torch.nn.GaussianNLLLoss #TODO



iteration = 0
for epoch in range(10):
    for i, data in enumerate(x):
        optimizer.zero_grad()
        label = y[i]
        label = label.unsqueeze(dim=0)
        data = data.unsqueeze(dim=0)
        loss = bayes_network.sample_elbo(inputs=data.to(device),
                        labels=label.to(device),
                        criterion=criterion,
                        sample_nbr=3,
                        complexity_cost_weight=1/1000)
        loss.backward()
        optimizer.step()
        
        iteration += 1
    print(epoch)

predicted = torch.zeros(x.shape[0])
for i, data in enumerate(x):
    predicted[i] = bayes_network(data.unsqueeze(dim=0))

predicted = predicted.detach()
plt.figure(0)
plt.scatter(x, y, s = 30, alpha = 1, marker = "o", color = 'red', label = 'Data')
plt.plot(x,y_target, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'Target function')
plt.plot(x, predicted.numpy(), color= 'cornflowerblue', alpha=0.8, linewidth = 3, label='learned model $\mu$')
plt.legend()
plt.show()