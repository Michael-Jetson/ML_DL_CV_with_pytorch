import torch
import torch.nn as nn
 
drop = nn.Dropout()
x = torch.ones(10)
  
# Train mode   
drop.train()
print(drop(x)) # tensor([2., 2., 0., 2., 2., 2., 2., 0., 0., 2.])   
  
# Eval mode   
drop.eval()
print(drop(x)) # tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
