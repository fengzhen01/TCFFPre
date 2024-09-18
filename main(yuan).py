import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.utils as np_utils
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import re

# VMD数据集
# VMD训练集
with open('../input/VMD-12/dA5_1.txt', 'r') as f1:   # 打开文件
    fid = list(f1)      # 将文件内容读入并转换成列表存储在fid变量中
VMD_1_3_data = [[0 for m in range(1)] for n in range(len(fid))]    # 初始化一个二维列表，每个内部列表包含一个元素0，外部列表长度与fid相同
for i in range(len(fid)):
    n = fid[i]       # n保存文件中第i行的数据
    a = n.strip()    # 去除该行的尾部换行符'\n'
    # b= n.replace('\r','').replace('\n','').replace('\t','').replace('.','')    # 去除改行的'\n'
    # 以空格分割改行数据，保存数据在b列表中
    t = 0
    for j in range(len(a)):
        VMD_1_3_data[i].insert(t, float(a[j]))     # 将每个字符转换为浮点数，并插入到对应的子列表中
        t = t + 1
    VMD_1_3_data[i].pop(3)    # 移除每行数据的第四个元素
print(VMD_1_3_data[1])

with open('../input/VMD-12/dA5_2.txt', 'r') as f2:  # 打开文件
    fid = list(f2)  # 文件转成列表
VMD_2_3_data = [[0 for m in range(1)] for n in range(len(fid))]
for i in range(len(fid)):
    n = fid[i]       # n保存文件第i行数据
    a = n.strip()    # 去除改行的'\n'
    t = 0
    for j in range(len(a)):
        VMD_2_3_data[i].insert(t, float(a[j]))
        t = t + 1
    VMD_2_3_data[i].pop(3)
# print(VMD_2_3_data[1])

with open('../input/VMD-12/dA5_IE_(3).txt', 'r') as f3:  # 打开文件
    fid = list(f3)  # 文件转成列表
VMD_3_3_data = [[0 for m in range(1)] for n in range(len(fid))]
for i in range(len(fid)):
    n = fid[i]  # n保存文件第i行数据
    a = n.strip()  # 去除改行的'\n'
    t = 0
    for j in range(len(a)):
        VMD_3_3_data[i].insert(t, float(a[j]))
        t = t + 1
    VMD_3_3_data[i].pop(3)

with open('../input/VMD-12/dA5_IE_(4).txt', 'r') as f4:  # 打开文件
    fid = list(f4)  # 文件转成列表
VMD_4_3_data = [[0 for m in range(1)] for n in range(len(fid))]
for i in range(len(fid)):
    n = fid[i]  # n保存文件第i行数据
    a = n.strip()  # 去除改行的'\n'
    t = 0
    for j in range(len(a)):
        VMD_4_3_data[i].insert(t, float(a[j]))
        t = t + 1
    VMD_4_3_data[i].pop(3)

with open('../input/VMD-12/dA5_IE_(5).txt', 'r') as f5:  # 打开文件
    fid = list(f5)  # 文件转成列表
VMD_5_3_data = [[0 for m in range(1)] for n in range(len(fid))]
for i in range(len(fid)):
    n = fid[i]  # n保存文件第i行数据
    a = n.strip()  # 去除改行的'\n'
    t = 0
    for j in range(len(a)):
        VMD_5_3_data[i].insert(t, float(a[j]))
        t = t + 1
    VMD_5_3_data[i].pop(3)

VMD_3_17500 = (VMD_1_3_data[0:3500] + VMD_2_3_data[0:3500] + VMD_3_3_data[0:3500] + VMD_4_3_data[0:3500] + VMD_5_3_data[0:3500])
np.random.seed(12345)     # 将五个数据列表的前3500行合并,设置随机数种子，确保每次随机结果一致
np.random.shuffle(VMD_3_17500)    # 打乱合并后的数据顺序
# print(VMD_3_17500)
train_VMD_data_tensor = tf.convert_to_tensor(VMD_3_17500)     # 将数据转换成TensorFlow张量
train_VMD_data_tensor_reshape = tf.reshape(train_VMD_data_tensor, [17500, 3, 1])    # 将张量重新塑形为[17500, 3, 1]的形状

VMD_3_7500 = (VMD_1_3_data[3500:5000] + VMD_2_3_data[3500:5000] + VMD_3_3_data[3500:5000] + VMD_4_3_data[3500:5000])
np.random.seed(12345)
np.random.shuffle(VMD_3_7500)
test_VMD_data_tensor = tf.convert_to_tensor(VMD_3_7500)
test_VMD_data_tensor_reshape = tf.reshape(test_VMD_data_tensor, [7500, 3, 1])

# ITD数据集
# ITD训练集
with open('../input/itd-feature/dA5_1.txt', 'r') as f1:     # 打开文件
    fid = list(f1)      # 将文件内容读入并转换成列表存储在fid变量中
ITD_1_3_data = [[0 for m in range(1)] for n in range(len(fid))]    # 初始化一个二维列表，每个内部列表包含一个元素0，外部列表长度与fid相同
for i in range(len(fid)):
    n = fid[i]     # n保存文件第i行数据
    t = 0
    for j in range(len(a)):
        ITD_1_3_data[i].insert(t, float(a[j]))     # 将每个字符转换为浮点数，并插入到对应的子列表中
        t = t + 1
    ITD_1_3_data[i].pop(3)    # 移除每行数据的第四个元素
print(ITD_1_3_data[1])

with open('../input/itd-feature/dA5_2.txt', 'r') as f2:  # 打开文件
    fid = list(f2)  # 文件转成列表
ITD_2_3_data = [[0 for m in range(1)] for n in range(len(fid))]
for i in range(len(fid)):
    n = fid[i]  # n保存文件第i行数据
    a = n.strip()  # 去除改行的'\n'
    t = 0
    for j in range(len(a)):
        ITD_2_3_data[i].insert(t, float(a[j]))
        t = t + 1
    ITD_2_3_data[i].pop(3)
# print(ITD_2_3_data[1])

with open('../input/itd-feature/dA5_3.txt', 'r') as f3:  # 打开文件
    fid = list(f3)  # 文件转成列表
ITD_3_3_data = [[0 for m in range(1)] for n in range(len(fid))]
for i in range(len(fid)):
    n = fid[i]  # n保存文件第i行数据
    a = n.strip()  # 去除改行的'\n'
    t = 0
    for j in range(len(a)):
        ITD_3_3_data[i].insert(t, float(a[j]))
        t = t + 1
    ITD_3_3_data[i].pop(3)

with open('../input/itd-feature/dA5_4.txt', 'r') as f4:  # 打开文件
    fid = list(f4)  # 文件转成列表
ITD_4_3_data = [[0 for m in range(1)] for n in range(len(fid))]
for i in range(len(fid)):
    n = fid[i]  # n保存文件第i行数据
    a = n.strip()  # 去除改行的'\n'
    t = 0
    for j in range(len(a)):
        ITD_4_3_data[i].insert(t, float(a[j]))
        t = t + 1
    ITD_4_3_data[i].pop(3)

with open('../input/itd-feature/dA5_5.txt', 'r') as f5:  # 打开文件
    fid = list(f5)  # 文件转成列表
ITD_5_3_data = [[0 for m in range(1)] for n in range(len(fid))]
for i in range(len(fid)):
    n = fid[i]  # n保存文件第i行数据
    a = n.strip()  # 去除改行的'\n'
    t = 0
    for j in range(len(a)):
        ITD_5_3_data[i].insert(t, float(a[j]))
        t = t + 1
    ITD_5_3_data[i].pop(3)

ITD_3_17500 = (ITD_1_3_data[0:3500] + ITD_2_3_data[0:3500] + ITD_3_3_data[0:3500] + ITD_4_3_data[0:3500] + ITD_5_3_data[0:3500])
np.random.seed(12345)    # 将五个数据列表的前3500行合并，设置随机数种子，确保每次随机结果一致
np.random.shuffle(ITD_3_17500)     # 打乱合并后的数据顺序
# print(ITD_3_17500)
train_ITD_data_tensor = tf.convert_to_tensor(ITD_3_17500)     # 将数据转换成TensorFlow张量
train_ITD_data_tensor_reshape = tf.reshape(train_ITD_data_tensor, [17500, 3, 1])     # 将张量重新塑形为[17500, 3, 1]的形状
ITD_3_7500 = (ITD_1_3_data[3500:5000] + ITD_2_3_data[3500:5000] + ITD_3_3_data[3500:5000] + ITD_4_3_data[3500:5000] + ITD_5_3_data[3500:5000])
np.random.seed(12345)      # 设置随机数种子，确保每次随机结果一致
np.random.shuffle(ITD_3_7500)    # 打乱合并后的数据顺序
test_ITD_data_tensor = tf.convert_to_tensor(ITD_3_7500)    # 将数据转换成TensorFlow张量
test_ITD_data_tensor_reshape = tf.reshape(test_ITD_data_tensor, [7500, 3, 1])     # 将张量重新塑形为[7500, 3, 1]的形状

# 训练集标签train_label
with open('../input/trainlabel/train_label_17500.txt', 'r') as f:
    f_label = list(f)    # 将文件内容读取为列表，每个元素为一行
train_label = [[0 for m in range(1)] for n in range(len(f_label))]
for i in range(len(f_label)):
    train_label[i] = f_label[i].strip()     # 移除每个标签的尾部空白字符
    train_label[i] = int(train_label[i])    # 将标签字符串转换为整数
np.random.seed(12345)     # 设置随机数种子，确保每次操作的可复现性
np.random.shuffle(train_label)    # 打乱标签的顺序，以保证数据随机性
train_label_tensor = tf.convert_to_tensor(train_label)    # 将标签列表转换为TensorFlow张量
train_label_tensor_numpy = train_label_tensor.numpy()     # 将张量转换回NumPy数组，以便进行进一步处理
train_label_tensor_OneHot = np_utils.to_categorical(train_label_tensor_numpy)    # 使用one-hot编码将标签转换为二进制矩阵形式
train_label_tensor_OneHot = tf.convert_to_tensor(train_label_tensor_OneHot)      # 将One-hot编码后的数据再次转换为TensorFlow张量
# 测试集标签test_label

with open('../input/testlabel/test_label_3500.txt', 'r') as f:
    f_label = list(f)
test_label = [[0 for m in range(1)] for n in range(len(f_label))]
for i in range(len(f_label)):
    test_label[i] = f_label[i].strip()
    test_label[i] = int(test_label[i])
np.random.seed(12345)
np.random.shuffle(test_label)
test_label_tensor = tf.convert_to_tensor(test_label)
test_label_tensor_numpy = test_label_tensor.numpy()
test_label_tensor_OneHot = np_utils.to_categorical(test_label_tensor_numpy)
test_label_tensor_OneHot = tf.convert_to_tensor(test_label_tensor_OneHot)

# 构建模型
# channel_VMD，VMD数据处理
inputs1 = tf.keras.Input(shape=(3, 1))     # 定义输入层，接收形状为(3, 1)的数据
conv1_1 = tf.keras.layers.Conv1D(filters=3, kernel_size=25, padding='same', activation='relu', input_shape=(3, 1))(
    inputs2)    # 卷积层，使用3个过滤器和大小为25的卷积核
pool1_1 = tf.keras.layers.AveragePooling1D(pool_size=1)(conv1_1)     # 池化层，使用平均池化
flatten1_1 = tf.keras.layers.Flatten()(pool1_1)     # 扁平化层，用于将多维输入一维化

# channel_VMD，VMD数据处理
inputs2 = tf.keras.Input(shape=(3, 1))     # 定义输入层，接收形状为(3, 1)的数据
conv2_1 = tf.keras.layers.Conv1D(filters=3, kernel_size=25, padding='same', activation='relu', input_shape=(3, 1))(
    inputs2)    # 卷积层，使用3个过滤器和大小为25的卷积核
pool2_1 = tf.keras.layers.AveragePooling1D(pool_size=1)(conv2_1)     # 池化层，使用平均池化
flatten2_1 = tf.keras.layers.Flatten()(pool2_1)     # 扁平化层，用于将多维输入一维化

# channel_ITD，ITD数据处理
inputs3 = tf.keras.Input(shape=(3, 1))
conv3_1 = tf.keras.layers.Conv1D(filters=3, kernel_size=25, padding='same', activation='relu', input_shape=(3, 1))(
    inputs3)
pool3_1 = tf.keras.layers.AveragePooling1D(pool_size=1)(conv3_1)
flatten3_1 = tf.keras.layers.Flatten()(pool3_1)
# FC_layers，合并层
merged = tf.keras.layers.concatenate([flatten1_1, flatten2_1, flatten3_1])   # 将三个通道的扁平化层输出合并

# 全连接层（FC Layers）
dense1 = tf.keras.layers.Dense(1024, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(merged)    # 全连接层，1024个神经元，激活函数为tanh，加L2正则化
dense_drop1 = tf.keras.layers.Dropout(0.3)(dense1)
dense2 = tf.keras.layers.Dense(512, activation='tanh', kernel_regularizer=regularizers.l2(0.005))(dense_drop1)    # 全连接层，512个神经元
dense3 = tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(0.005))(dense2)         # 全连接层，256个神经元
dense_drop2 = tf.keras.layers.Dropout(0.3)(dense3)
dense4 = tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.005))(dense_drop2)    # 全连接层，128个神经元
dense5 = tf.keras.layers.Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.005))(dense4)          # 全连接层，64个神经元
dense_drop3 = tf.keras.layers.Dropout(0.3)(dense5)
dense6 = tf.keras.layers.Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.005))(dense_drop3)     # 全连接层，32个神经元
dense7 = tf.keras.layers.Dense(16, activation='tanh', kernel_regularizer=regularizers.l2(0.005))(dense6)          # 全连接层，16个神经元
outputs = tf.keras.layers.Dense(5, activation='softmax')(dense7)     # 输出层，5个神经元，使用softmax激活函数进行多分类
model = tf.keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)    # 创建模型
# 下载之前保存的模型
# model=load_model('./model_save.h5')
# 打印模型的结构
model.summary()
# 编译模型
model.compile(loss="categorical_crossentropy",     # 使用分类交叉熵作为损失函数
              optimizer=keras.optimizers.Adadelta(0.1),    # 使用Adadelta优化器，学习率为0.1
              metrics=["accuracy"])                        # 评估模型性能的指标为准确率
history = model.fit([train_VMD_data_tensor_reshape, train_ITD_data_tensor_reshape],
                    train_label_tensor_OneHot,
                    epochs=30, batch_size=10, validation_split=0.2, shuffle=False)     # 训练30轮，每个批次10个样本，20%的数据用于验证，不打乱数据顺序
# #测试集测试模型
model.evaluate([test_VMD_data_tensor_reshape, test_ITD_data_tensor_reshape],
               test_label_tensor_OneHot)

# 获取训练过程中的准确率和损失值
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# 绘制训练和验证的准确率曲线
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
# 绘制训练和验证的损失曲线
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()    # 显示图表