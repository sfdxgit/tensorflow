# TFHub 迁移学习 transfer learning
#使用TFHub中的预训练模型ImageNet进行迁移学习，实现图像分类，数据集使用CIFAR-10
#TFHub上有很多预训练好的模型，ImageNet数据集大约有1500万张图片，2.2万类 图片固定大小（244，244，3）

#下载ImageNet Classifier
import numpy as np 
from PIL import Image
import matplotlib.pylab as plt
import tensorflow as tf 
import tensorflow_hub as hub 
from tensorflow.keras import layers,datasets
url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
model = tf.keras.Sequential([
    hub.KerasLayer(url,input_shape = (244,244,3))
])
#测试任意图片
tutu=tf.keras.utils.get_file('tutu.png','https://geektutu.com/img/icon.png')
tutu=Image.open(tutu).resize((224,224))
result = model.predict(np.array(tutu).reshape(1,224,224,3)/255.0)
ans = np.argmax(result[0],axis=-1)
print('result.shape:',result.shape,'ans:',ans)# result.shape: (1, 1001) ans: 332
labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',labels_url)
imagenet_labels = np.array(open(labels_path).read().splitlines())
print(imagenet_labels[ans])

#迁移学习
#resize数据集  ImageNet Classifier输入固定为（244，244，3） CIFAR-10数据集图片大小为（32，32）  
# 将32*32转为224*224   pillow库的resize
def resize(d,size=(224,224)):
    return np.array([np.array(Image.fromarray(v).resize(size,Image.ANTIALIAS))
        for i,v in enumerate(d)])
(train_x,train_y),(test_x,test_y) = datasets.cifar10.load_data()
train_x,test_x = resize(train_x[:30000])/255.0,resize(test_x)/255.0
train_y = train_y[:30000]#读取全部数据会撑爆内存，所以只截取30000张图片

#下载特征提取层  TFHub 提供了 ImageNet Classifier 去掉了最后的分类层的版本
feature_extractor_url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,input_shape=(224,224,3))
feature_extractor_layer.trainable = False#这一层的训练值保持不变
#添加分类层   （定义自己的输出层）
#在特征提取层后面，添加输出为10的全连接层，用于最后的分类
model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer=tf.keras.opentimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])
print(model.summary())

#训练并评估模型
history = model.fit(train_x,train_y,epochs=1)
loss,acc = model.evaluate(test_x,test_y)
print(acc)