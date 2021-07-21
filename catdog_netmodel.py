import torch.nn as nn

class CDNet(nn.Module):
    def __init__(self):
        super(CDNet,self).__init__()
        #卷积层
        self.convnet=nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1,bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
                )
        #全连接层
        self.linenet=nn.Sequential(
                nn.Linear(64*64*8,1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000,1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000,2),
                nn.Softmax(dim=1)
                )

    def forward(self,x):
        x=self.convnet(x)
        x=x.view(x.size(0),64*64*8)
        out=self.linenet(x)
        return out
