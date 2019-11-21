#使用迁移算法解决二分类问题----电影正向和负向评论分类
import numpy as np 
import tensorflow as tf 
import tensorflow_hub as hub 
import tensorflow_datasets as tfds


# print(hub.__version__)
train_validation_split = tfds.Split.TRAIN.subsplit([6,4])
(train_data,validation_data),test_data=tfds.load(
    name="imdb_reviews",
    split=(train_validation_split,tfds.Split.TEST),
    as_supervised=True
)

#数据格式：每个例子包含一句电影评论和对应的标签，0或1.
train_examples_batch,train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)

#搭建模型
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1" #从 TensorFlow Hub 中选用的pre-trained 文本嵌入模型 直接将影评文本转换为向量
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
    dtype=tf.string, trainable=True) #将句子转为向量
hub_layer(train_examples_batch[:3])
#搭建完整的神经网络模型
model=tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
#损失函数和优化器
model.compile(optimizer='adam',
    loss = 'binary_crossentropy',#binary_crossentropy更适合处理概率问题，mean_squared_error适合处理回归(Regression)问题
    metrics=['accuracy'])
#训练模型
history=model.fit(train_data.shuffle(10000).batch(512),
    epochs=20,
    validation_data=validation_data.batch(512),
    verbose=1)
#评估模型
results=model.evaluate(test_data.batch(512),verbose=0)
for name,value in zip(model.metrics_names,results):
    print("%s:%.3f" % (name,value))
    # loss: 0.314
    # accuracy: 0.866

