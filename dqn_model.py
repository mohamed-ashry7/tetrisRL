import torch
import torch.nn as nn 
import numpy as np



class DQN(nn.Module):
    
    
    def __init__(self,input_shape,number_actions):
        super(DQN,self).__init__()
        input_shape=np.prod(input_shape)
        
        self.fc_layers=nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(160, 110),
            nn.BatchNorm1d(110),
            nn.ReLU(),
            nn.Linear(110, 70),
            nn.BatchNorm1d(70),
            nn.ReLU(),
            nn.Linear(70, number_actions)
            
        )

        
    
    def flat_features_number(self,x):
        features_size=x.size()[1:]
        return np.prod(features_size)
    
    def forward(self,x):
        return self.fc_layers(x.view(-1,self.flat_features_number(x)))