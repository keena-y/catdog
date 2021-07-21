import torch
import numpy as np
import os
from natsort import natsorted
import cv2

data_test = np.load('catdog_test_set.npy',  allow_pickle=True)
net = torch.load('catdog_net3.pkl')



img_path = 'sample'

img_datas = []
image_list = os.listdir(img_path)

cat_list = []
dog_list = []

for img_name in image_list:
    if img_name[0:3] == 'cat':
        cat_list.append(img_name)
    else:
        dog_list.append(img_name)
cat_list = natsorted(cat_list)
dog_list = natsorted(dog_list)

test_catdog_list = cat_list[70:100] + dog_list[70:100]


#cat_test = []
count = 0
for img_name in test_catdog_list:
    image = cv2.imread(img_path + '/' + img_name)
    image = cv2.resize(image, (128, 128))
    x = np.zeros((1, 3, 128, 128))
    x[0, 0, :, :] = image[:, :, 2] / 256.
    x[0, 1, :, :] = image[:, :, 1] / 256.
    x[0, 2, :, :] = image[:, :, 0] / 256.
    x = torch.tensor(x, dtype=torch.float32)
    out = net(x)
    maxnum = torch.max(out, 1)[1]
    #计算由net分类结果与真实分类一样的个数
    if maxnum == 0:
        if img_name[0:3] == 'cat':
            count += 1
    else:
        if img_name[0:3] == 'dog':
            count += 1

print(count/60)

