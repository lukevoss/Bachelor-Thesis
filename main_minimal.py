# %%
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Externals_minimal import nishDataset, postprocessing, DoseRNN
import os
import time
import scipy.io

def train(model, train_loader, batch_size, n_step, imsize, criterion, optimizer, epoch, num_epochs, device):
  
  allLoss = []
  
  model.train()

  for ii, data in enumerate(train_loader):
      img, dose = data
      img = Variable(img.view(batch_size * n_step, -1)).to(device)
      dose = Variable(dose.view(-1,1,imsize,imsize)).to(device)
      # ===================forward=====================
      lstm = model(img)
      output = model.backend(lstm)
      output = output.view(-1,1,imsize,imsize) #changes back to 15 x 15
      loss = criterion(output, dose)
      # ===================backward====================
      for param in model.parameters():
        param.grad = None
      loss.backward()
      optimizer.step()
      
      allLoss.append(loss.item())
      
  # ===================log========================
  print('epoch [{}/{}], loss:{:.9f}'
        .format(epoch+1, num_epochs, np.sum(allLoss)/len(allLoss)))
  
  return allLoss, img, dose, output.cpu().data

def test(model, test_loader, batch_size, n_step, imsize, criterion, device):
  
  allLoss = []
  model.eval()
  with torch.no_grad():
    for ii, data in enumerate(test_loader):
      
      img , dose = data
      img = Variable(img.view(batch_size * n_step, -1)).to(device)
      dose = Variable(dose.view(-1,1,imsize,imsize)).to(device)
      # ===================forward=====================
      lstm = model(img)
      output = model.backend(lstm)
      output = output.view(-1,1,imsize,imsize)
      loss = criterion(output, dose)
      allLoss.append(loss.item())
    
    print('test loss:{:.9f}'.format(np.sum(allLoss)/len(allLoss)))
    

  return allLoss, img, dose, output.cpu().data

def prepare_directory():
  # %% Preparing the folder for the output

  if not os.path.exists('./out'):
    os.mkdir('./out')

  dirnum = 13 # attempt number, to avoid over writing data
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


  batch_size = 2
  n_step = 80 # how long the sequence is
  imsize = 15 # imsize * imsize is the size of each slice in the sequence

  num_epochs = 10
  learning_rate =  1e-5

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

    model = DoseRNN(batch_size, n_neuron, n_step, imsize, n_layer).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    tt = time.time()
    for epoch in range(num_epochs):
      loss, img_t,dose_t, output_t= train(model, train_loader, batch_size, n_step, imsize, criterion, optimizer, epoch, num_epochs, device)
      loss_train.append(loss) 
      #loss_train.append(loss.detach) #TODO Speed up?
      loss, img, dose, output = test(model, test_loader, batch_size, n_step, imsize, criterion, device)
      loss_test.append(loss) 
      #loss_test.append(loss.detach) #TODO Speed up?

    print('elapsed time: {}'.format(time.time() - tt))
  except KeyboardInterrupt:

    pass
  postprocessing(model, loss_train, loss_test, description, 'LSTM', 'modelarch')

  # %% plotting some results

  nplot = 10 # number of test data to plot

  for i in range(nplot):
    
    img , dose = iter(test_loader).next()

    img = Variable(img.view(batch_size * n_step, -1)).to(device)
    dose = Variable(dose.view(-1,1,imsize,imsize)).to(device)

    lstm = model(img)
    output = model.backend(lstm)
    output = output.view(-1,1,imsize,imsize)
    output = output.cpu().data

    npimg = np.array(img.cpu().view(-1,imsize,imsize))
    npdose = np.array(dose.cpu()).squeeze(axis = 1)
    npoutput = np.array(output).squeeze(axis = 1)
    
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

