'''
Created on 2024年10月9日

@author: Administrator
'''

import torch.nn as nn
import torch.nn.init as init

class PointTransform(nn.Module):
    def __init__(self, medium=3)->None:
        nn.Module.__init__(self)
        
        self.trans1 = nn.Linear(3, medium)
        self.trans2 = nn.Linear(medium, 3)

        self.init_parameters()
        
    def forward(self, x):
        x = self.trans1(x)
        x = self.trans2(x)
        
        return x
    
    def init_parameters(self):
        init.eye_(self.trans1.weight)
        init.ones_(self.trans1.bias)
        
        init.eye_(self.trans2.weight)
        init.ones_(self.trans2.bias)
    
if __name__ == '__main__':
    pass