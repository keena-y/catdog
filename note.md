# 猫狗识别
### `sample`文件放所有图片
### `catdog_build_dataset.py`读取图片，划分数据集(训练集和测试集）
> 可以根据自己项目需要修改路径`img_path`、图片大小、训练集和测试集比例、分类标签
### `catdog_netmodel.py` 构建模型
### `catdog_train.py` 训练模型，画出模型损失函数图，并将训练好的模型保存（`catdog_net3.pkl`）
> 可以根据自己项目需要选择优化器`optimizer`、修改迭代次数`epoch`、batch参数`samplenum、minibatch、picsize`
### `catdog_realuse.py` 利用已训练好的模型对测试集分类
### `catdog_getacc.py` 计算准确率 ![alt 属性文本](https://github.com/keena-y/yolov3-imgs/blob/main/13.jpg)