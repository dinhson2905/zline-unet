from unet import UNet
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(1,3,572,572)
x_train = x.to(device)

net = UNet(3,1)
net.to(device)
out = net(x_train)

print(out)