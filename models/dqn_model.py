import torch
import torch.nn as nn 
import numpy as np


from models.noisy_layer import NoisyLinear


class DQN(nn.Module):
    
    
    def __init__(self,input_shape,number_actions):
        super(DQN,self).__init__()
        self.input_shape=input_shape

        self.common_layers=nn.Sequential(
            NoisyLinear(input_shape, 47),
            nn.BatchNorm1d(47),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            NoisyLinear(47, 71),
            nn.BatchNorm1d(71),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            
        )
        self.value_layers=nn.Sequential(
            NoisyLinear(71, 87),
            nn.BatchNorm1d(87),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            NoisyLinear(87, number_actions)
            
        )
        self.adv_layers=nn.Sequential(
            NoisyLinear(71, 87),
            nn.BatchNorm1d(87),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            NoisyLinear(87, 1)
            
        )

        
    def forward(self,x):
        val,adv = self.val_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


    def val_adv(self,x):
        x=self.common_layers(x.view(-1,self.input_shape))
        return self.value_layers(x) ,self.adv_layers(x) 