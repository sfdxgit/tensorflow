#保存与加载模型
# 使用 tf.keras接口训练、保存、加载模型，数据集选用 MNIST

#准备数据
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets,layers,models,callbacks
from tensorflow.keras.datasets import mnist 

import os
file_path = os.path.abspath('./mnist.npz')

(train_x,train_y),(test_x,test_y) = datasets.mnist.load_data(path=file_path)
train_y,test_y = train_y[:1000],test_y[:1000]
train_x = train_x[:1000].reshape(-1,28*28)/255.0
test_x = test_x[:1000].reshape(-1,28*28)/255.0

#搭建模型
def create_model():
    model = models.Sequential([
        layers.Dense(512,activation='relu',input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer='adam',metrics=['accuracy'],
        loss='sparse_categorical_crossentropy')
    return model
def evaluate(target_model):
    _, acc = target_model.evaluate(test_x,test_y)
    print('Restore model,accuracy:{:5.2f}%'.format(100*acc))

#自动保存checkpoints
    #这样做，一是训练结束后得到了训练好的模型，使用时不必再重新训练，
    # 二是训练过程被中断，可以从断点继续训练
    # 设置tf.keras.callbacks.ModelCheckpoint回调可以实现这一点
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = callbacks.ModelCheckpoint(
    checkpoint_path,verbose=1,save_weights_only=True,period=10)#period=10 每10epoch保存一次
model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_x,train_y,epochs=50,callbacks=[cp_callback],
    validation_data=(test_x,test_y),verbose=0)

#加载权重
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)
evaluate(model)
#手动保存权重
model.save_weights('./checkpoints/mannul_checkpoint')
model = create_model()
model.load_weights('./checkpoints/mannul_checkpoint')
evaluate(model)

#保存整个模型
# 上面的示例仅仅保存了模型中的权重(weights)，
# 模型和优化器都可以一起保存，
# 包括权重(weights)、模型配置(architecture)和优化器配置(optimizer configuration)。
# 这样做的好处是，当你恢复模型时，完全不依赖于原来搭建模型的代码。
    #直接调用model.save即可保存HDF5格式的文件
model.save('my_model.h5')
    #从HDF5中恢复完整的模型
new_model = models.load_model('my_model.h5')
evaluate(new_model)

    #保存为saved_model格式
import time
saved_model_path = "./saved_models/{}".format(int(time.time()))
tf.keras.experimental.export_saved_model(model,saved_model_path)
    #恢复模型并预测
new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
print(model.predict(test_x).shape) 
#saved_model格式的模型可以直接用来预测，但是saved_model没有保存优化器配置
#如果要使用evaluate方法，需要先compile
new_model.compile(optimizer=model.optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
evaluate(new_model)