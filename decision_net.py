import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat,rearrange

class DecisionNet(nn.Module):
    def __init__(self, num_inputs=800,patch_size=16,frame_size=4
                ):
        super(DecisionNet, self).__init__()
        self.patch_size = patch_size
        self.frame_size = frame_size
       
        self.conv1 = nn.Conv3d(num_inputs, num_inputs // 2, kernel_size=(1,3,3), padding=(0,1,1))
        self.maxpool1 = nn.MaxPool3d((1,2,2))
        self.conv2 = nn.Conv3d(num_inputs // 2, num_inputs // 4, kernel_size=(1,3,3), padding = (0,1,1))
        self.maxpool2 = nn.MaxPool3d((1,2,2))
        self.conv3 = nn.Conv3d(num_inputs // 4, num_inputs // 2, kernel_size=(3,3,3), stride=1)
        self.maxpool3 = nn.MaxPool3d((2,2,2))
        self.conv4 = nn.Conv3d(num_inputs // 2, num_inputs, kernel_size=1, stride=1)
        # # self.conv4 = nn.Conv3d(32, 32, 3, stride=2, padding=1)

        # self.linear = nn.Linear(3136, 512)
        # self.linear2 = nn.Linear(512, 1)
    def forward(self, x):
        x = rearrange(x, 'b n (d w h) -> b n d w h', w = self.patch_size, h = self.patch_size, d = self.frame_size)
        x = F.leaky_relu(self.maxpool1(self.conv1(x)), inplace=False)
        x = F.leaky_relu(self.maxpool2(self.conv2(x)), inplace=False)
        x = F.leaky_relu(self.maxpool3(self.conv3(x)), inplace=False)
   
        x = self.conv4(x)
    
        # x4 = F.relu(self.conv4(x3), inplace=False)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(x)

if __name__ == '__main__':
    x = torch.randn(1,800,16*16*4)
    num_inputs = x.shape[1]
    net = DecisionNet(num_inputs=num_inputs)

    # print(net(x).shape)
    print(net(x).shape,net(x).max(),net(x).min())