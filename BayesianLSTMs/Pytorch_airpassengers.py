import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

#Import
training_set = pd.read_csv('airline-passengers.csv')#in current working directory

training_set = training_set.iloc[:,1:2].values


#create sequenced data
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
    x = torch.tensor(x).float() #Solved: Hier lag Fehler
    y = torch.tensor(y).float() #Solved: Hier lag Fehler
    return x,y

#preprocessing data
sc = StandardScaler()
training_data = sc.fit_transform(training_set)

seq_length = 4
x, y = sliding_windows(training_data, seq_length)

#70% Training 30% Testing
train_size = int(len(y) * 0.7)
test_size = len(y) - train_size

trainX, testX, trainY, testY = train_test_split(x,
                                                    y,
                                                    test_size=.3,
                                                    random_state=42,
                                                    shuffle=False)#Solved: Hier lag Fehler
# dataX = Variable(torch.Tensor(np.array(x)))
# dataY = Variable(torch.Tensor(np.array(y)))

# trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
# trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

# testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
# testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

months = np.arange(start=0, stop=len(y),step = 1)
train_months = np.arange(start=0,stop=train_size,step=1)
test_months = np.arange(start=train_size, stop=train_size+test_size, step=1)

ds = torch.utils.data.TensorDataset(trainX, trainY)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=True)

#build model
@variational_estimator           
class BayesianNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(BayesianNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        #TODO num_layers
        self.lstm = BayesianLSTM(input_size, hidden_size)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        
        # Propagate input through LSTM
        lstm_out, (h_out, _) = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        #lstm_out, self.hidden = self.lstm(x)
        #h_out = h_out.view(-1, self.hidden_size)

        #out = self.fc(h_out)
        out = self.fc(lstm_out)
        return out

# Training
num_epochs = 1500
learning_rate = 0.001

input_size = 1
hidden_size = 2
num_layers = 1

num_classes = 1

bayesian_network = BayesianNN(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(bayesian_network.parameters(), lr=learning_rate)


# Train the model
for epoch in range(num_epochs):
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        # obtain the loss function
        loss = bayesian_network.sample_elbo(inputs=datapoints,
                            labels=labels,
                            criterion=criterion, 
                            sample_nbr=7,
                            complexity_cost_weight=1/x.shape[0])
        loss.backward()
        
        optimizer.step()

    if epoch%250==0:
            preds_test = bayesian_network(testX)[:,0].unsqueeze(1)
            loss_test = criterion(preds_test, testY)
            print("Iteration: {} Val-loss: {:.4f}".format(str(epoch), loss_test))

bayesian_network.eval()


dataY_plot = y.data.numpy()
dataY_plot = sc.inverse_transform(dataY_plot)

plt.plot(dataY_plot, linestyle = 'dashed', color = 'black', linewidth = 3, label = 'Ground Truth')
plt.scatter(months,dataY_plot, s = 30, alpha = 1, marker = "o", color = 'red', label = 'Data')
plt.axvline(x=train_size, c='r', linestyle='--')

ensemble_size = 20

test_predict = [bayesian_network(testX) for i in range(ensemble_size)]
for i in range(ensemble_size):
    prediction = test_predict[i] 
    prediction= prediction.data.numpy()
    prediction = sc.inverse_transform(prediction)
    test_predict[i] = prediction
test_predict_mean =np.stack(test_predict)
means = test_predict_mean.mean(axis=0)
stds = test_predict_mean.std(axis=0)
y_upper = means + (2 * stds)
y_lower = means - (2 * stds)
# for i in range(ensemble_size):
#     prediction = bayesian_network(testX)
#     #prediction = prediction[:, 0, :]
#     prediction= prediction.data.numpy()
#     prediction = sc.inverse_transform(prediction)
#     plt.plot(test_months, prediction, color= 'cornflowerblue', alpha=0.8, linewidth = 3, label='Learned Model')
plt.fill_between(test_months, y_upper[:,0], y_lower[:,0], alpha = 0.4, color='skyblue', label='Standard Deviation')



plt.plot(test_months,means, label = 'Prediction')
plt.suptitle('Time-Series Prediction')
plt.legend()
plt.show()