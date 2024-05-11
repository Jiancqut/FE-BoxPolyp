import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from lib.pvtv2 import pvt_v2_b2
from .decoder import CASCADE
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

logger = logging.getLogger(__name__)

def np2th(weights, conv=False):

    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
class Net(nn.Module):
    def __init__(self, n_class=1):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = pvt_v2_b2() 
        path = '/home/ubuntu/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
     
        self.decoder = CASCADE(channels=[512, 320, 128, 64])
        
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        
    
        if x.size()[1] == 1:
            x = self.conv(x)
       
        x1, x2, x3, x4 = self.backbone(x)
        

        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1]) 
 
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        p=p1+p2+p3+p4

        return p


