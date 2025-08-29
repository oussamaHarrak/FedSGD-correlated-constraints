import torch.nn as nn

class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 4),  
            nn.Tanh(),             
            nn.Linear(4, 2) 
        )
        #nn.Linear(28 * 28, 32),
            
        #nn.ReLU(),
        #nn.Linear(32, 2)  # single output

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.fc(x) 