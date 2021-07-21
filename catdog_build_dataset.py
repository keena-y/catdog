import cv2
import numpy as np
import os
from natsort import natsorted

img_path = 'sample'#路径
img_datas = []
image_list = os.listdir(img_path)#返回指定路径下的文件和文件夹列表。
# image_list = natsorted(image_list)
#print(image_list)
cat_list = []
dog_list = []
#将猫和狗的图片分类
for img_name in image_list:
    if img_name[0:3] == 'cat':
        cat_list.append(img_name)
    else:
        dog_list.append(img_name)
cat_list = natsorted(cat_list)#natsorted排序函数
dog_list = natsorted(dog_list)


#生成猫的训练集和测试集
cat_train = []
cat_test = []
#0-69为训练集
for img_name in cat_list[0:70]:
    image = cv2.imread(img_path + '/' + img_name)#读取图片，输出BGR像素储存向量 h*w*3 h个w*3的三维数组
    image = cv2.resize(image, (128, 128))#修改图片尺寸,得到128*128*3的数组
    cat_train.append(image.reshape(3, 128, 128))#改变维度为3*128*128数组

#70-99为测试集
for img_name in cat_list[70:100]:
    image = cv2.imread(img_path + '/' + img_name)
    image = cv2.resize(image, (128, 128))
    cat_test.append(image)

#生成狗的训练集和测试集
dog_train = []
dog_test = []

for img_name in dog_list[0:70]:
    image = cv2.imread(img_path + '/' + img_name)
    image = cv2.resize(image, (128, 128))
    dog_train.append(image.reshape(3, 128, 128))

for img_name in dog_list[70:100]:
    image = cv2.imread(img_path + '/' + img_name)
    image = cv2.resize(image, (128, 128))
    dog_test.append(image)


catdog_train_set = cat_train + dog_train #猫狗训练集汇总  140*3*128*128
cat = 0
dog = 1
catdog_train_label = [cat] * 70 + [dog] * 70#分类标签


catdog_test_set = cat_test + cat_train #猫狗测试集汇总   60*3*128*128

#保存数组信息，生成.npy文件
np.save('catdog_train_set', catdog_train_set)
np.save('catdog_test_set', catdog_test_set)
np.save('catdog_train_label', catdog_train_label)
# np.save('dog_train', dog_train)
# np.save('dog_test', dog_test)