import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class nishDataset(Dataset):
    def __init__(self, inp, out, split, group = ''):
        
        # Load the data
        x = torch.load(inp).type(torch.FloatTensor)
        y = torch.load(out).type(torch.FloatTensor)
        
        split = int(split * x.shape[0])
        
        print('Whole dataset shape: CT input: {}, Dose Output: {}'.format(x.shape, y.shape))
        
        if group == 'train':
            x = x[:split]
            y = y[:split]
            print('Trainset shape:      CT input: {}, Dose Output: {}'.format(x.shape, y.shape))
        elif group == 'test':
            x = x[split:]
            y = y[split:]
            print('Testset shape:       CT input: {}, Dose Output: {}'.format(x.shape, y.shape))
        
        
        self.X = x
        self.y = y
        
    def __getitem__(self, index):
        local_x = self.X[index]
        local_y = self.y[index]
        return local_x, local_y
    
    def __len__(self):
        return len(self.X)
    


class DoseRNN(nn.Module):
    def __init__(self, batch_size, n_neuron, nstep, imsize, nlayer, dropout = 0):
        super(DoseRNN, self).__init__()
        
        self.n_neurons = n_neuron
        self.batch_size = batch_size
        self.n_steps = nstep
        self.n_inputs = imsize**2
        self.n_outputs = imsize**2
        self.n_layers = nlayer
        self.flat = batch_size * nstep * self.n_inputs
        #self.rnn = nn.RNN(self.n_inputs, self.n_neurons, self.n_layers, nonlinearity = 'relu', dropout = dropout)
        self.lstm = nn.LSTM(self.n_inputs, self.n_neurons, self.n_layers, dropout = dropout)

        self.backend = nn.Sequential(
            nn.Linear(self.n_neurons, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, self.n_outputs)
            )


    def init_hidden(self,):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(self.n_layers, self.batch_size, self.n_neurons)).to(device)
    

    def forward(self, x): # x.shape to be ==> n_steps X batch_size X n_inputs
        
        x = x.view((self.n_steps, self.batch_size, self.n_inputs))
        # self.hidden = self.init_hidden()
        
        #lstm_out, self.hidden = self.rnn(x)#, self.hidden)
        lstm_out, self.hidden = self.lstm(x)
        
        return lstm_out.view(-1, self.n_neurons)

        
def postprocessing(model, loss_train, loss_test, description, stage, ii = 0):
  
  loss_train = np.array(loss_train)
  loss_test = np.array(loss_test)
  
  save_filename = './out/ae_{}/conv_{}_{}.pth'.format(description,stage, ii)
  torch.save(model.state_dict(), save_filename)
  save_loss = './out/ae_{}/loss_train_{}_{}.npy'.format(description, stage, ii) 
  np.save(save_loss, loss_train)
  save_loss = './out/ae_{}/loss_test_{}_{}.npy'.format(description, stage, ii)
  np.save(save_loss, loss_test)
  sum_loss_train = np.sum(loss_train, axis = 1)/loss_train.shape[1]
  sum_loss_test = np.sum(loss_test, axis = 1)/loss_test.shape[1]
  fig, ax = plt.subplots(figsize = (8,8));
  ax.plot(np.arange(loss_train.shape[0]), sum_loss_train, label = 'Train')
  ax.plot(np.arange(loss_test.shape[0]), sum_loss_test, label = 'Test')
  ax.legend()
  save_filename = './out/ae_{}/Loss_{}_{}.png'.format(description, stage, ii)
  plt.savefig(save_filename)
  plt.close()
  