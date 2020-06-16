import torch
import torch.nn as nn 
import numpy as np



class DQN(nn.Module):
    
    
    def __init__(self,input_shape,number_actions):
        super(DQN,self).__init__()
        input_shape=np.prod(input_shape)
#         self.conv_layers=nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
        
#         linear_input_shape=self._conv_to_linear_shape(input_shape)
#         print(linear_input_shape)
        self.fc_layers=nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 160),
            nn.ReLU(),
            nn.Linear(160, 110),
            nn.ReLU(),
            nn.Linear(110, 70),
            nn.ReLU(),
            nn.Linear(70, number_actions),
            nn.ReLU()
            
        )

        
#     def _conv_to_linear_shape(self,shape):
#         o=self.conv_layers(torch.zeros(1,*shape))
#         return int(np.prod(o.size()))
    
    def flat_features_number(self,x):
        features_size=x.size()[1:]
        return np.prod(features_size)
    
    def forward(self,x):
#         x=self.conv_layers(x)
        return self.fc_layers(x.view(-1,self.flat_features_number(x)))