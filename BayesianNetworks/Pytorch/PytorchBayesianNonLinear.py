import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


NUM_POINTS = 1000
x = torch.linspace(-5, 2, NUM_POINTS)
y_target = 4 * x * np.cos(np.pi * np.sin(x)) + 1
y = y_target + torch.randn(x.shape[0]) * 0.5
x= x.unsqueeze(dim=1)
y= y.unsqueeze(dim=1)
input_size = 1
output_size = 1 #2 TODO 

@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, 128, prior_sigma_1 = 1, prior_sigma_2 = 1, prior_pi=2) 
        self.blinear2 = BayesianLinear(128, 64, prior_sigma_1 = 1, prior_sigma_2 = 1, prior_pi=2)
        self.blinear3 = BayesianLinear(64, 32, prior_sigma_1 = 1,prior_sigma_2 = 1, prior_pi=2)
        self.blinear4 = BayesianLinear(32, 16, prior_sigma_1 = 1, prior_sigma_2 = 1, prior_pi=2)
        self.blinear5 = BayesianLinear(16, output_dim, prior_sigma_1 = 1,prior_sigma_2 = 1, prior_pi=2)
        
    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.silu(x_)
        x_ = self.blinear2(x_)
        x_ = F.silu(x_)
        x_ = self.blinear3(x_)
        x_ = F.silu(x_)
        x_ = self.blinear4(x_)
        x_ = F.silu(x_)
        return self.blinear5(x_)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bayes_network = BayesianRegressor(input_size, output_size).to(device)
optimizer = optim.Adam(bayes_network.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()#TODO

ds = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(ds, batch_size=50, shuffle=True)
epochs = 200

for epoch in range(epochs):
    for i, (datapoints,labels) in enumerate(dataloader):
        optimizer.zero_grad()

        loss = bayes_network.sample_elbo(inputs=datapoints.to(device),
                        labels=labels.to(device),
                        criterion=criterion, 
                        sample_nbr=1,
                        complexity_cost_weight=1/x.shape[0])
        loss.backward()
        optimizer.step()
    print("Epoch %d/%d" % (epoch+1, epochs))


ensemble_size = 100
preds = [bayes_network(x.to(device)) for i in range(ensemble_size)]
preds_mean = torch.stack(preds)
means = preds_mean.mean(axis=0).cpu().detach().numpy()
stds = preds_mean.std(axis=0).cpu().detach().numpy()
y_upper = means + (2 * stds)
y_lower = means - (2 * stds)
plt.figure(0)
plt.scatter(x, y, s = 30, alpha = 1, marker = "o", color = 'red', label = 'Data')
plt.plot(x,y_target, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'Target function')
#for i in range(ensemble_size):
    #plt.plot(x, preds[i].cpu().detach().numpy(), color= 'blue', alpha=0.8, linewidth = 3)
plt.plot(x, means, color= 'cornflowerblue', alpha=0.8, linewidth = 3, label='learned model $\mu$')
plt.fill_between(x[:,0], y_upper[:,0], y_lower[:,0], alpha = 0.4, color='skyblue', label='Standard Deviation')
plt.legend()
plt.show()
