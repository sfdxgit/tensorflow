#利用CNN提高手写数字识别Mnist 的准确性

#CNN模型代码(train.py)
import os
import tensorflow as tf
from tensorflow.keras import datasets,layers,models

class CNN(object):
    def __init__(self):
        model = models.Sequential()
        #第一层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
        model.add(layers.MaxPooling2D((2,2)))
        #第二层卷积，卷积核大小3*3，64个
        model.add(layers.Conv2D(64,(3,3),activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        #第三层卷积，卷积核大小3*3，64个
        model.add(layers.Conv2D(64,(3,3),activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dense(10,activation='softmax'))#10位恰好可以表示0-9十个数字
        model.summary()
        self.model = model

#mnist数据集预处理
class DataSource(object):
    def __init__(self):
        #mnist数据集存储的位置，如果不存在将自动下载
        data_path = os.path.abspath(os.path.dirname(__file__))+'C:/Users/dell/mnist.npz'
        (train_images,train_labels),(test_images,test_labels) = datasets.mnist.load_data(path=data_path)
        #6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000,28,28,1))
        test_images = test_images.reshape((10000,28,28,1))
        #像素值映射到0-1之间
        train_images,test_images = train_images/255.0,test_images/255.0
        self.train_images,self.train_labels = train_images,train_labels
        self.test_images,self.test_labels = test_images,test_labels
    
#开始训练并保存训练结果
class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()
    def train(self):
        check_path = 'D:/ckpt/cp-{epoch:04d}.ckpt'
        #period每隔5epoch保存一次
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,verbose=1,period=5)
        self.cnn.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_images,self.data.train_labels,epochs=5,callbacks=[save_model_cb])
        test_loss,test_acc = self.cnn.model.evaluate(self.data.test_images,self.data.test_labels)
        print("准确率：%.4f,共测试了%d张图片"%(test_acc,len(self.data.test_labels)))
    
if __name__=="__main__":
    app = Train()
    app.train()

#为了将模型的训练和加载分开，预测代码写在predict.py中
#图片预测（predict.py）
import tensorflow as tf 
from PIL import Image
import numpy as np 
from train import CNN

class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('D:/ckpt')
        self.cnn=CNN()
        self.cnn.model.load_weights(latest)
    def predict(self,image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, (28, 28, 1))
        x = np.array([1 - flatten_img])
        # API refer: https://keras.io/models/model/
        y = self.cnn.model.predict(x)
        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(y[0])
        print('        -> Predict digit', np.argmax(y[0]))

if __name__ == "__main__":
    app = Predict()
    app.predict('../test_images/0.png')
    app.predict('../test_images/1.png')
    app.predict('../test_images/4.png')