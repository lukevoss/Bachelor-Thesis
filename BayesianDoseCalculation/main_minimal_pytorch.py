# %%
import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Externals_minimal_pytorch import nishDataset, postprocessing, BayesianDoseLSTM
import os
import time
import scipy.io


def train(bayesian_network, train_loader, batch_size, n_step, imsize, criterion, optimizer, epoch, num_epochs, device):
  
  allLoss = []
  
  bayesian_network.train()

  for ii, (img, dose) in enumerate(train_loader):
      
      optimizer.zero_grad()
      img = Variable(img.view(batch_size,n_step,imsize*imsize)).to(device)
      dose = Variable(dose.view(batch_size,n_step,imsize*imsize)).to(device)
      # ===================forward=====================
      #output = bayesian_network(img)
      loss = bayesian_network.sample_elbo(inputs=img.to(device),
                               labels=dose.to(device),
                               criterion=criterion,
                               sample_nbr=3,
                               complexity_cost_weight=1/(10000))
      #loss = criterion(output, dose)
      # ===================backward====================
      for param in bayesian_network.parameters():
        param.grad = None
      loss.backward()
      optimizer.step()
      
      allLoss.append(loss.item())
      
  # ===================log========================
  print('epoch [{}/{}], loss:{:.9f}'
        .format(epoch+1, num_epochs, np.sum(allLoss)/len(allLoss)))
  
  return allLoss

def test(bayesian_network, test_loader, batch_size, n_step, imsize, criterion, device, length_data):
  
  allLoss = []
  bayesian_network.eval()
  with torch.no_grad():
    for ii, data in enumerate(test_loader):
      
      img , dose = data
      img = Variable(img.view(batch_size,n_step,imsize*imsize)).to(device)
      dose = Variable(dose.view(batch_size,n_step,imsize*imsize)).to(device)
      # ===================forward=====================
      output = bayesian_network(img)
      loss = criterion(output, dose)
      allLoss.append(loss.item())
    
    print('test loss:{:.9f}'.format(np.sum(allLoss)/len(allLoss)))
    

  return allLoss

def prepare_directory():
  # %% Preparing the folder for the output

  if not os.path.exists('./out'):
    os.mkdir('./out')

  dirnum = 3 # attempt number, to avoid over writing data
  description = 'attempt{}'.format(dirnum)

  # creating the output folder
  rootdir = './out/ae_{}'.format(description)
  if os.path.exists(rootdir):
    input('Are you sure you want to overwrite?')
  else:
      os.mkdir(rootdir)


  # copying a version of the code in destination folder
  import shutil
  shutil.copyfile('./main_minimal.py', './out/ae_{}/{}'.format(description,'inp.py'))
  shutil.copyfile('./Externals_minimal.py', './out/ae_{}/{}'.format(description,'Ext.py'))
  return description, rootdir


def main():
  
  description, rootdir = prepare_directory()

  # %%  loading the data set
  params = {
      'inp' : './datasets/inp_box801515.pt', #changed
      'out' : './datasets/out_box801515.pt', #changed
      'split' : .8  
      }


  batch_size = 32
  n_step = 80 # how long the sequence is
  imsize = 15 # imsize * imsize is the size of each slice in the sequence

  num_epochs = 200
  starting_learning_rate =  1e-3

  n_layer = 1 # number of layers in LSTM/RNN
  n_neuron = 1000 # number of neurons in LSTM/RNN

  dd_train = nishDataset(**params, group = 'train')
  dd_test = nishDataset(**params, group = 'test')

    
  train_loader = torch.utils.data.DataLoader(dd_train, shuffle = True, batch_size=batch_size, drop_last = True, pin_memory=False)
  test_loader = torch.utils.data.DataLoader(dd_test, shuffle = True, batch_size=batch_size, drop_last = True, pin_memory=False)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # %% Training happens here

  try:
    loss_train, loss_test = [], []

    bayesian_network = BayesianDoseLSTM(batch_size, n_neuron, n_step, imsize, n_layer).to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(bayesian_network.parameters(), lr = starting_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.98)
    tt = time.time()
    
    for epoch in range(num_epochs):
      loss = train(bayesian_network, train_loader, batch_size, n_step, imsize, criterion, optimizer, epoch, num_epochs, device) #output_t gel??scht
      loss_train.append(loss) 
      loss = test(bayesian_network, test_loader, batch_size, n_step, imsize, criterion, device, len(dd_test))
      loss_test.append(loss) 
      
      scheduler.step()
    print('elapsed time: {}'.format(time.time() - tt))
  except KeyboardInterrupt:

    pass
  postprocessing(bayesian_network, loss_train, loss_test, description, 'LSTM', 'bayesian_networkarch')

  # %% plotting some results
  bayesian_network.eval()
  nplot = 20 # number of test data to plot
  ensemble_size = 30

  for i in range(nplot):
    img , dose = iter(test_loader).next()
    img = img[0, ...]#take first batch
    dose = dose[0, ...]
    img = Variable(img.view(1,n_step,imsize*imsize)).to(device)
    dose = Variable(dose).to(device)
    npoutput = [np.array(bayesian_network(img).view(n_step,imsize,imsize).cpu().detach()) for i in range(ensemble_size)]
    img = img.view(n_step, imsize, imsize)
    npimg = np.array(img.cpu())
    npdose = np.array(dose.cpu())
    #npoutput = np.array(output.cpu().data)
    slc = int(np.floor(imsize/2))
    if not os.path.exists(rootdir):
      os.mkdir(rootdir)
    
    output_mean =np.stack(npoutput)
    means = output_mean.mean(axis=0)
    stds = output_mean.std(axis=0)

    
    
    fig, axs = plt.subplots(3,1, figsize = (20,10))
    fig.suptitle('Dose Estimation using ANN')
    image_1 = axs[0].imshow(np.transpose(npimg[:,slc,:]))
    axs[0].set_title('CT input, Max ED = {}'.format(npimg.max()))
    fig.colorbar(image_1, ax = axs[0], fraction=0.046, pad=0.04)
    image_2 = axs[1].imshow(np.transpose(npdose[:,slc,:]))
    axs[1].set_title('MC dose, Dose Integral = {}'.format(npdose.sum()))
    fig.colorbar(image_2, ax = axs[1], fraction=0.046, pad=0.04)
    image_3 = axs[2].imshow(np.transpose(means[:,slc,:]))
    axs[2].set_title('Estimated dose, Dose Integral = {}'.format(means.sum()))
    fig.colorbar(image_3, ax = axs[2], fraction=0.046, pad=0.04)
    plt.savefig('{}/2d_{}_{}_{}.png'.format(rootdir, i, starting_learning_rate, n_layer))
    plt.close()
  
      
    savedict = {
    'ct' : npimg,
    'dose' : npdose,
    'dose_ann' : npoutput}
    scipy.io.savemat('{}/dosecubes_{}.mat'.format(rootdir,i), savedict)     

if __name__ == "__main__":
  main()        

