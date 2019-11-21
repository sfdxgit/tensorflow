# MNIST 图像分类 
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape) # (60000, 28, 28)
print(test_images.shape)
print(train_images.shape[0])
# 预处理  图片的每个像素值在0-255之间，需要转为0-1
train_images = train_images / 255.0
test_images = test_images / 255.0
# 搭建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#将输入从28x28 的二维数组转为784的一维数组
    keras.layers.Dense(128, activation='relu'), #全连接层
    keras.layers.Dense(10, activation='softmax') #经过 softmax 后，返回了和为1长度为10的概率数组，每一个数分别代表当前图片属于分类0-9的概率
])
# 编译模型
model.compile(optimizer='adam', #优化器算法，更新模型参数的算法
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型 训练神经网络
model.fit(train_images, train_labels, epochs=10)
# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)
#预测
predictions = model.predict(test_images)
print(predictions[0])

print(np.argmax(predictions[0])) # 9  argmax取得最大值的下标
print(test_labels[0]) #9