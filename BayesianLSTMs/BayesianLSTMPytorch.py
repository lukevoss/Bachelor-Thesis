#import pandas as pd
from cmath import pi
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from collections import deque

NUM_POINTS_SEQUENCE = 1000
x = torch.linspace(0, 20*2*pi, NUM_POINTS_SEQUENCE)
y_target = x*np.sin(x) + x*x*0.1
y = y_target + torch.randn(x.shape[0]) * 2
input_size = 1
output_size = 1
minima = np.arange(start=0,stop=20*2*pi,step=2*pi)
y_minima = minima*np.sin(minima) + minima*minima*0.1 

x = torch.zeros(NUM_POINTS_SEQUENCE,minima.shape[0])
y = torch.zeros(NUM_POINTS_SEQUENCE,minima.shape[0])
y_target = torch.zeros(NUM_POINTS_SEQUENCE,minima.shape[0])
for i, start_sequence in enumerate(minima):
    stop_sequence = minima[i+1]
    x[:,i] = torch.linspace(start_sequence, stop_sequence, NUM_POINTS_SEQUENCE)
    y_target = x*np.sin(x) + x*x*0.1
    y = y_target + torch.randn(x.shape[0]) * 2
    

@variational_estimator
class BayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.lstm_1 = BayesianLSTM(input_dim, 20, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.linear = nn.Linear(20, output_dim)
            
    def forward(self, x):
        x_, _ = self.lstm_1(x)
        
        #gathering only the latent end-of-sequence for the linear layer
        #x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bayes_network = BayesianNN(input_size, output_size).to(device)
optimizer = optim.Adam(bayes_network.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()#TODO

ds = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

epochs = 10
for epoch in range(epochs):
    for i, (datapoints, labels) in enumerate(dataloader):
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
plt.plot(minima,y_minima, marker='o', color = 'green', label = 'minima')
# plt.plot(x, means, color= 'cornflowerblue', alpha=0.8, linewidth = 3, label='learned model $\mu$')
# plt.fill_between(x[:,0], y_upper[:,0], y_lower[:,0], alpha = 0.4, color='skyblue', label='Standard Deviation')
plt.legend()
plt.show()






# #pred_unscaled
# original = close_prices_unscaled[1:][window_size:]
# df_pred = pd.DataFrame(original)
# df_pred["Date"] = df.Date
# df["Date"] = pd.to_datetime(df_pred["Date"])
# df_pred = df_pred.reset_index()
# #df_pred = df_pred.set_index('Date')
# def pred_stock_future(X_test,
#                                            future_length,
#                                            sample_nbr=10):
    
#     #sorry for that, window_size is a global variable, and so are X_train and Xs
#     global window_size
#     global X_train
#     global Xs
#     global scaler
    
#     #creating auxiliar variables for future prediction
#     preds_test = []
#     test_begin = X_test[0:1, :, :]
#     test_deque = deque(test_begin[0,:,0].tolist(), maxlen=window_size)

#     idx_pred = np.arange(len(X_train), len(Xs))
    
#     #predict it and append to list
#     for i in range(len(X_test)):
#         #print(i)
#         as_net_input = torch.tensor(test_deque).unsqueeze(0).unsqueeze(2)
#         pred = [net(as_net_input).cpu().item() for i in range(sample_nbr)]
        
        
#         test_deque.append(torch.tensor(pred).mean().cpu().item())
#         preds_test.append(pred)
        
#         if i % future_length == 0:
#             #our inptus become the i index of our X_test
#             #That tweak just helps us with shape issues
#             test_begin = X_test[i:i+1, :, :]
#             test_deque = deque(test_begin[0,:,0].tolist(), maxlen=window_size)

#     #preds_test = np.array(preds_test).reshape(-1, 1)
#     #preds_test_unscaled = scaler.inverse_transform(preds_test)
    
#     return idx_pred, preds_test
# def get_confidence_intervals(preds_test, ci_multiplier):
#     global scaler
    
#     preds_test = torch.tensor(preds_test)
    
#     pred_mean = preds_test.mean(1)
#     pred_std = preds_test.std(1).detach().cpu().numpy()

#     pred_std = torch.tensor((pred_std))
#     #print(pred_std)
    
#     upper_bound = pred_mean + (pred_std * ci_multiplier)
#     lower_bound = pred_mean - (pred_std * ci_multiplier)
#     #gather unscaled confidence intervals

#     pred_mean_final = pred_mean.unsqueeze(1).detach().cpu().numpy()
#     pred_mean_unscaled = scaler.inverse_transform(pred_mean_final)

#     upper_bound_unscaled = upper_bound.unsqueeze(1).detach().cpu().numpy()
#     upper_bound_unscaled = scaler.inverse_transform(upper_bound_unscaled)
    
#     lower_bound_unscaled = lower_bound.unsqueeze(1).detach().cpu().numpy()
#     lower_bound_unscaled = scaler.inverse_transform(lower_bound_unscaled)
    
#     return pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled
# future_length=7
# sample_nbr=4
# ci_multiplier=5
# idx_pred, preds_test = pred_stock_future(X_test, future_length, sample_nbr)
# pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled = get_confidence_intervals(preds_test,
#                                                                                           ci_multiplier)
# y = np.array(df.Close[-750:]).reshape(-1, 1)
# under_upper = upper_bound_unscaled > y
# over_lower = lower_bound_unscaled < y
# total = (under_upper == over_lower)