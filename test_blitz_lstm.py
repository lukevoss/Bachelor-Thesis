import torch
import torch.nn as nn

criterion = nn.MSELoss()
first = torch.ones(2,4,4)
first[1,...]=first[1,...]*2
second = torch.ones(2,4,4)
second[1,...]=first[1,...]*3
second[1,...]=first[1,...]*4
loss = criterion(first, second)
print(loss)
first.view(-1,16)
second.view(-1,16)
loss_new=criterion(first,second)
print(loss)