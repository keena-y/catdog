from catdog_netmodel import CDNet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#build net
net=CDNet()

#read x y
x=np.load('catdog_train_set.npy')/255. #数据进行标准化处理[0,1]之间
x=torch.tensor(x,dtype=torch.float32)
y=np.load('catdog_train_label.npy')
y=torch.tensor(y,dtype=torch.long)

#optimizer优化器  随机梯度下降
opt=torch.optim.SGD(net.parameters(),lr=0.03)

#loss 交叉熵损失函数
loss_func=nn.CrossEntropyLoss()

##batch
samplenum=140#训练集样本总数
minibatch=35
picsize=128


##loop
losslist=[]
#迭代次数
for epoch in range(100):
    print(epoch)
    for i in range(int(samplenum/minibatch)):
        x00=x[i*minibatch:(i+1)*minibatch:,:,:]
        y00=y[i*minibatch:(i+1)*minibatch]
        out=net(x00)
        loss=loss_func(out,y00)#计算损失函数
        opt.zero_grad()
        loss.backward()#更新模型的梯度，包括 weights 和 bias
        opt.step()
    losslist.append(loss.item())
    #print('epoch:%d, loss:%4.3f'%(epoch,loss))

#save net
torch.save(net,'catdog_net3.pkl') #保存训练后的网络

losslist=np.array(losslist)#100*1数组
plt.plot(losslist)
plt.title('loss')
plt.savefig('catdog_loss.jpg',dpi=256)
plt.close()
