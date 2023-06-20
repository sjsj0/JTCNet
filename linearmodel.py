import torch
import torch.nn as nn

# m = nn.Linear(2, 1)
# input = torch.randn(32, 2)
# output = m(input)
# print(output.size())

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super(MLP,self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(640,320),
      nn.ReLU(),
      # nn.Linear(1280, 2560),
      # nn.ReLU(),
      # nn.Linear(2560, 1280),
      # nn.ReLU(),
      # nn.Linear(1280, 640),
      # nn.ReLU(),
      # nn.Linear(640, 320),
      # nn.ReLU(),
      nn.Linear(320, 64),
      nn.ReLU(),
      nn.Linear(64,1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)


# m = MLP()
# input = torch.randn(32, 2)
# output = m(input)
# print(output.size())