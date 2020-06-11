import torch
import torch.nn as nn 
import numpy as np



class DQN(nn.Module):
    
    
    def __int__(self,input_shape,number_actions):
        super(DQN,self).__init__()
        
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        linear_input_shape=self._conv_to_linear_shape(input_shape)
        
        self.fc_layers=nn.Sequential(
            nn.Linear(linear_input_shape, 256),
            nn.ReLU().
            nn.Linear(256, number_actions)
            
        )

        
    def _conv_to_linear_shape(self,shape):
        o=self.conv_layers(torch.zeros(1,*shape))
        return int(np.prod(o.size))
    
    
    def forward(self,x):
        x=self.conv_layers(x)
        return self.fc_layers(x.view(x.size()[0],-1))