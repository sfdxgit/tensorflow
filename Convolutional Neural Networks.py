#卷积神经网络（CNN）分类  CIFAR-10 
#CNN模型代码
#主要使用Keras.layers提供的Conv2D（卷积）与MaxPooling2D（池化）函数
#CNN的输入是维度为 (image_height, image_width, color_channels)的张量
    #黑白图片：color_channels=1 
    #彩色图片：RGB 4个通道(R,G,B,A)，A代表透明度，取值范围是0-1
    # 通过参数input_shape传给网络的第一层

#cNN模型代码
# import os
# import tensorflow as tf 
# from tensorflow.keras import datasets,layers,models

# class CNN(object):
#     def __init__(self):
#         model = models.Sequential()
#         #第一层卷积，卷积核大小为3*3，32个 28*28为待训练图片大小
#         model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
#         model.add(layers.MaxPooling2D((2,2)))
#         #第二层卷积，卷积核大小为3*3，64个
#         model.add(layers.Conv2D(64,(3*3),activation='relu'))
#         model.add(layers.MaxPooling2D((2,2)))
#         #第三层卷积，卷积核大小为3*3,64个
#         model.add(layers.Conv2D(64,(3,3),activation='relu'))

#         model.add(layers.Flatten())#用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
#         model.add(layers.Dense(64,activation='relu'))#全连接层
#         model.add(layers.Dense(10,activation='softmax'))#全连接层
#         print(model.summary())#用来打印我们定义的模型的结构
#         self.model = model

#CIFAR-10 60000张彩色图片，共10类 每张图片的大小是 32x32x3  R/G/B取值范围是0-255
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import layers,datasets,models

(train_x,train_y),(test_x,test_y) = datasets.cifar10.load_data()
#展示前十五张图片
plt.figure(figsize=(5,3))
plt.subplots_adjust(hspace=0.1)
for n in range(15):
    plt.subplots_adjust(3,5,n+1)
    plt.imshow(train_x[n])#负责对图像进行处理，并显示其格式
    plt.axis('off')
_ = plt.suptitle('CIFAR-10 Example')
#将0-255的像素值转换到0-1
train_x,test_x = train_x/255.0,test_x/255.0
print('train_x shape:',train_x.shape,'test_x shape:',test_x.shape)# (50000, 32, 32, 3), (10000, 32, 32, 3)

#卷积层
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))#最大池化即选择图像区域的最大值作为该区域池化后的值，另一个常见的池化操作是平均池化，即计算图像区域的平均值作为该区域池化后的值
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
print(model.summary())

#全连接层
#CIFAR-10数据集：目的是对图像进行分类，即期望输出一个长度为10的一维向量
#第k个值代表输入图片分类为k的概率
#因此需要Dense层，将3维的卷积层输出，转换为1维
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
#编译训练模型
model.complie(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(train_x,train_y,epochs=5)
#评估模型
test_loss,test_acc=model.evaluate(test_x,test_y)
print(test_acc) #0.683

#结论：CNN非常适合用来处理图像，这个模型用来训练MINIST手写数字数据集，能达到99%的正确率
#训练CIFAR10数据集，只有68.3%的正确率，可以使用负责网络模型或者迁移学习来提高准确率