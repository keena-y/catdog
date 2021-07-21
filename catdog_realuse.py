import numpy as np
import cv2
import torch


##read image
picnum=np.random.randint(100)
pictype='cat' if np.random.randint(2)==0 else 'dog'
picpath='sample'
picname='%s.%d.jpg'%(pictype,picnum)

img=cv2.imread('%s\\%s'%(picpath,picname))
img=cv2.resize(img,(128,128))
x=np.zeros((1,3,128,128))#生成一个1*3*128*128的零矩阵
#将BGR各像素向量标准化处理后放入x数组中
x[0,0,:,:]=img[:,:,2]/256.
x[0,1,:,:]=img[:,:,1]/256.
x[0,2,:,:]=img[:,:,0]/256.
x=torch.tensor(x,dtype=torch.float32)

##read net
net=torch.load('catdog_net3.pkl')

##classification
out=net(x)#1*2矩阵

maxnum=torch.max(out,1)[1]#得出分类结果  第1个tensor是每行最大值对应的标签组成的
if maxnum==0:
    print('%s___real: %s, net: cat'%(picname,pictype))
else:
    print('%s___real: %s, net: dog'%(picname,pictype))

cv2.imwrite('clas-img.jpg',img)#保存图片

