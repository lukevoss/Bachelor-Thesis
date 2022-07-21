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


def train(bayesian_network, train_loader, batch_size, n_step, imsize, criterion, optimizer, epoch, num_epochs, device, length_data):
  
  allLoss = []
  
  bayesian_network.train()

  for ii, (img, dose) in enumerate(train_loader):
      
      
      img = Variable(img.view(batch_size,n_step,imsize*imsize)).to(device)
      dose = Variable(dose.view(batch_size,n_step,imsize*imsize)).to(device)
      # ===================forward=====================
      output = bayesian_network(img)
      loss = bayesian_network.sample_elbo(inputs=output.to(device),
                               labels=dose.to(device),
                               criterion=criterion,
                               sample_nbr=3,
                               complexity_cost_weight=1/length_data)
      
      # ===================backward====================
      for param in bayesian_network.parameters():
        param.grad = None
      loss.backward()
      optimizer.step()
      
      allLoss.append(loss.item())
      
  # ===================log========================
  print('epoch [{}/{}], loss:{:.9f}'
        .format(epoch+1, num_epochs, np.sum(allLoss)/len(allLoss)))
  
  return allLoss, img, dose, output.cpu().data

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
      loss = bayesian_network.sample_elbo(inputs=output.to(device),
                               labels=dose.to(device),
                               criterion=criterion,
                               sample_nbr=3,
                               complexity_cost_weight=1/length_data)
      allLoss.append(loss.item())
    
    print('test loss:{:.9f}'.format(np.sum(allLoss)/len(allLoss)))
    

  return allLoss, img, dose, output.cpu().data

def prepare_directory():
  # %% Preparing the folder for the output

  if not os.path.exists('./out'):
    os.mkdir('./out')

  dirnum = 12 # attempt number, to avoid over writing data
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


  batch_size = 1
  n_step = 80 # how long the sequence is
  imsize = 15 # imsize * imsize is the size of each slice in the sequence

  num_epochs = 5
  learning_rate =  1e-5

  n_layer = 1 # number of layers in LSTM/RNN
  n_neuron = 1000 # number of neurons in LSTM/RNN

  dd_train = nishDataset(**params, group = 'train')
  dd_test = nishDataset(**params, group = 'test')

  # ds = torch.utils.data.TensorDataset(x, y)
  # ds = torch.utils.data.TensorDataset(x, y)
    
  train_loader = torch.utils.data.DataLoader(dd_train, shuffle = True, batch_size=batch_size, drop_last = True, pin_memory=False)
  test_loader = torch.utils.data.DataLoader(dd_test, shuffle = True, batch_size=batch_size, drop_last = True, pin_memory=False)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # %% Training happens here

  try:
    loss_train, loss_test = [], []

    bayesian_network = BayesianDoseLSTM(batch_size, n_neuron, n_step, imsize, n_layer).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(bayesian_network.parameters(), lr = learning_rate)
    tt = time.time()
    
    for epoch in range(num_epochs):
      loss, img_t,dose_t, output_t= train(bayesian_network, train_loader, batch_size, n_step, imsize, criterion, optimizer, epoch, num_epochs, device, len(dd_train))
      loss_train.append(loss) 
      #loss_train.append(loss.detach) #TODO Speed up?
      loss, img, dose, output = test(bayesian_network, test_loader, batch_size, n_step, imsize, criterion, device, len(dd_test))
      loss_test.append(loss) 
      #loss_test.append(loss.detach) #TODO Speed up?

    print('elapsed time: {}'.format(time.time() - tt))
  except KeyboardInterrupt:

    pass
  postprocessing(bayesian_network, loss_train, loss_test, description, 'LSTM', 'bayesian_networkarch')

  # %% plotting some results

  nplot = 10 # number of test data to plot

  for i in range(nplot):
    
    img , dose = iter(test_loader).next()
    img = img[0, ...]#take first batch
    dose = dose[0, ...]
    img = Variable(img.view(1,n_step,imsize*imsize)).to(device)
    dose = Variable(dose).to(device)

    output = bayesian_network(img)
    output = output.view(n_step,imsize,imsize)
    img = img.view(n_step, imsize, imsize)

    npimg = np.array(img.cpu())
    npdose = np.array(dose.cpu())
    npoutput = np.array(output.cpu().data)
    
    slc = int(np.floor(imsize/2))
    
    if not os.path.exists(rootdir):
      os.mkdir(rootdir)
    fig, axs = plt.subplots(3,1, figsize = (20,10))
    fig.suptitle('Dose Estimation using ANN')
    image_1 = axs[0].imshow(np.transpose(npimg[:,slc,:]))
    axs[0].set_title('CT input, Max ED = {}'.format(npimg.max()))
    fig.colorbar(image_1, ax = axs[0], fraction=0.046, pad=0.04)
    image_2 = axs[1].imshow(np.transpose(npdose[:,slc,:]))
    axs[1].set_title('MC dose, Dose Integral = {}'.format(npdose.sum()))
    fig.colorbar(image_2, ax = axs[1], fraction=0.046, pad=0.04)
    image_3 = axs[2].imshow(np.transpose(npoutput[:,slc,:]))
    axs[2].set_title('Estimated dose, Dose Integral = {}'.format(npoutput.sum()))
    fig.colorbar(image_3, ax = axs[2], fraction=0.046, pad=0.04)
    plt.savefig('{}/2d_{}_{}_{}.png'.format(rootdir, i, learning_rate, n_layer))
    plt.close()
  
      
    savedict = {
    'ct' : npimg,
    'dose' : npdose,
    'dose_ann' : npoutput}
    scipy.io.savemat('{}/dosecubes_{}.mat'.format(rootdir,i), savedict)     

if __name__ == "__main__":
  main()        

