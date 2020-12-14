import os
import torch
from unet import UNet as _UNet

def unet_stomatch(pretrained=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = _UNet(n_channels=3, n_classes=1, bilinear=True)
    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, 'checkpoints/CP_epoch5.pth')
        state_dict = torch.load(checkpoint)
        net.load_state_dict(state_dict)
        # net.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    
    return net