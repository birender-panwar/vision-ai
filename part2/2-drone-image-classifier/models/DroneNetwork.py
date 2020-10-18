import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResizingConvNetwork(nn.Module):
    def __init__(self):
        super(ResizingConvNetwork, self).__init__()

        self.resizing224 = nn.Sequential(
            # None               
        )

        self.resizing448 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False)    #OUT: 224X224               
        )

        self.resizing672 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=3, bias=False)    #OUT: 224X224               
        )

        self.resizing896 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 448X448 
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False)    #OUT: 224X224              
        )

        self.resizing1344 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 672X672 
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=3, bias=False)    #OUT: 224X224              
        )

        self.resizing1792 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 896X896 
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 448X448     
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False)    #OUT: 224X224                   
        )

        self.resizing2688= nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 1344X1344 
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 672X672 
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=3, bias=False)    #OUT: 224X224               
        )

        self.resizing3584 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 1792X1792   
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 896X896 
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False),   #OUT: 448X448     
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=2, bias=False)    #OUT: 224X224         
        )

    def forward(self, x):
        #print(x.shape)
        imgsize = x.shape[2]

        # Apply Conv based on image sizes
        if (imgsize == 448):
            x = self.resizing448(x)
        elif (imgsize == 672):
            x = self.resizing672(x)
        elif (imgsize == 896):
            x = self.resizing896(x)
        elif (imgsize == 1344):
            x = self.resizing1344(x)
        elif (imgsize == 1792):
            x = self.resizing1792(x)
        elif (imgsize == 2688):
            x = self.resizing2688(x)
        elif (imgsize == 3584):
            x = self.resizing3584(x)
          
        return x

class DroneNetwork(nn.Module):
    def __init__(self, preTrainedNet):
        super(DroneNetwork, self).__init__()
        self.resizingNet = ResizingConvNetwork()
        self.preTrainedNet = preTrainedNet
        
    def forward(self, x):
        x = self.resizingNet(x)   # reszing to 224X224 through Learning
        x = self.preTrainedNet(x)
        return x
