#结构化数据分类  二分类（预测一个病人是否有心脏病）
# 1使用pandas加载csv文件
# 2使用tf.data打乱数据并获取batch
# 3使用特征工程将csv中的列映射为特征值  特征工程目的是最大限度的从原始数据中提取特征以供算法和模型使用。
# 4使用Keras搭建，训练和评估模型

#注：小数据集，建议使用决策树或者随机森林
import numpy as numpy 
import pandas as pd 
import tensorflow as tf 

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
#读取数据
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()
#分割训练集、验证集、测试集
train,test = train_test_split(dataframe,test_size = 0.2)
train,val = train_test_split(train,test_size = 0.2)
print(len(train),'train examples')#193
print(len(val),'validation examples')#49
print(len(test),'test examples')#61

#创建imput pipeline   如果是一个非常大的 CSV 文件，不能直接放在内存中，就必须直接使用 tf.data 从磁盘中直接读取数据了
#帮助函数，返回tf.data数据集
def df_to_dataset(dataframe,shuffle=True,batch_size=32):
    dataframe=dataframe.copy()
    labels = dataframe.pop('target')#pop可以将所选列从原数据块中弹出，原数据块不再保留该列
    ds=tf.data.Dataset.from_tensor_slices((dict(dataframe),labels))
    if shuffle:
        ds=ds.shuffle(buffer_size=len(dataframe))#将数据打乱的混乱程度shuffle用来打乱数据集，即混洗
    ds=ds.batch(batch_size)#从数据集中取出数据集个数
    return ds
batch_size=5
train_ds = df_to_dataset(train,batch_size=batch_size)
val_ds = df_to_dataset(val,shuffle=False,batch_size=batch_size)
test_ds = df_to_dataset(test,shuffle=False,batch_size=batch_size)
#理解pipeline
for feature_batch,label_batch in train_ds.take(1):
    print('Every feature:',list(feature_batch.keys()))
    print('A batch of ages:',feature_batch['age'])
    print('A batch of targets:',label_batch)

# #特征列示例
# example_batch = next(iter(train_ds))[0]
# #帮助函数
# def demo(feature_column):
#     feature_layer = layers.DenseFeatures(feature_column)
#     print(feature_layer(example_batch).numpy())
#     #特征列的输出是模型的输入  1、numeric_column 数值本身代表某个特征真实的值，因此转换后，值不发生改变
# age = feature_column.numeric_column("age")
# demo(age)
#     #2、用 bucketized_column 将年龄划分到不同的 bucket 中，使用独热编码
# age_buckets = feature_column.bucketized_column(age,boundaries=[18,25,30,40,45,50,55,60])
# demo(age_buckets)
#     #3、字符串不能直接传给模型。所以我们要先将字符串映射为数值,
#     #可以使用categorical_column_with_vocabulary_list 和 categorical_column_with_vocabulary_file 来转换，前者接受一个列表作为输入，后者可以传入一个文件
# thal = feature_column.categorical_column_with_vocabulary_list(
#     'thal',['fixed','normal','reversible'])
# thal_one_hot = feature_column.indicator_column(thal)
# demo(thal_one_hot)
#     #4、假设某一列有上千种类别，用独热编码来表示就不太合适了。这时候，可以使用 embedding column
#     # embedding column 可以压缩维度，因此向量中的值不再只由0或1组成，可以包含任何数字
#     # 在有很多种类别时使用 embedding column 是最合适的 最终的输出向量定长为8
# #embedding column的输入是categorical column
# thal_embedding = feature_column.embedding_column(thal,dimension=8)
# demo(thal_embedding)
#     #5、表示类别很多，使用 categorical_column_with_hash_bucket。
#     # 这个特征列会计算输入的哈希值，然后根据哈希值对字符串进行编码。哈希桶(bucket)个数即参数hash_bucket_size
# thal_hashed = feature_column.categorical_column_with_hash_bucket(
#     'thal',hash_bucket_size = 1000)
# demo(thal_hashed)
#     #6、将几个特征组合成一个特征，即 feature crosses，模型可以对每一个特征组合学习独立的权重
# crossed_feature = feature_column.crossed_column([age_buckets,thal],hush_bucket_size=1000)
# demo(feature_column.indicator_column(crossed_feature))

#选择需要使用的列
feature_columns = []
age = feature_column.numeric_column("age")
# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))
# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)
# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)
# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)
# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

#创建特征层
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
#创建新的pipeline，编译，训练模型
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128,activation = 'relu'),
    layers.Dense(128,activation = 'relu'),
    layers.Dense(1,activation = 'sigmoid')
])
model.compile(optimizer = 'adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
    run_eagerly=True)
model.fit(train_ds,
    validation_data=val_ds,
    epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
