import torch
import torch.nn as nn 
from linearmodel import MLP

GPU='0'
if torch.cuda.is_available():
    device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu') #
    print('on GPU')
else:
    print('on CPU')

# mlp = MLP()
# loss_function_mlp = nn.L1Loss().to(device)
# optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-4)
# mlp.to(device=device)

# label = torch.rand((16,1)).to(device=device)

# for i in range(100):
#     mlp.train()
#     inputData = torch.rand((16,2)).to(device=device)
#     optimizer_mlp.zero_grad()
#     MLP_HR_pr = mlp(inputData)

#     MLP_loss = loss_function_mlp(MLP_HR_pr,label)
#     MLP_loss.backward()
#     optimizer_mlp.step()
#     print(MLP_loss)

X = torch.rand((16,128))
Y = torch.rand((16,512))

Z = torch.cat((X,Y),1)
print(Z.shape)
print(Z[0].shape)
print(Z[8][500])
print(Y[8][372])
