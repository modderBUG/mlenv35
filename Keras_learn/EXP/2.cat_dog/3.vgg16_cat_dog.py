from keras.applications.vgg16 import VGG16,preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense
from keras.optimizers import Adam
import os

# 载入vgg模型 不包含全连接层
# !wget -nc "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" -P "/root/.keras/models"
model =VGG16(weights = 'imagenet',include_top=False)
model.summary()

datagen = ImageDataGenerator(
    rotation_range=40,      # 随即旋转角度
    width_shift_range=0.2,  # 随即水平平移
    height_shift_range=0.2, # 随机是指平移
    rescale=1./255,         # 数值归一化
    shear_range=0.2,        # 随即裁剪
    zoom_range=0.2,         # 随机放大
    horizontal_flip=True,   # 水平翻转
    fill_mode='nearest'     # 填充方式
)

batch_size = 32

train_steps= int((2000+batch_size-1)/batch_size)*10
test_steps =int((1000+batch_size-1)/batch_size)*10

train_generator  = datagen.flow_from_directory(
    'img/train',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode=None,    # 不生成标签
    shuffle=False       # 不随机打乱
)

# 得到训练数据集
bottleneck_features_train = model.predict_generator(generator=train_generator,train_steps=train_steps)
np.save(open('bottleneck_features_train.npy','wb',bottleneck_features_train))

# 测试数据
test_generator = datagen.flow_from_directory(
    'img/test',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode=None,  # 不生成标签
    shuffle=False  # 不随机打乱
)
bottleneck_features_test = model.predict_generator(generator=train_generator,train_steps=train_steps)
np.save(open('bottleneck_features_test.npy','wb',bottleneck_features_test))


train_data = np.load(open('bottleneck_features_train.npy','rb'))
labels =np.array([0]*1000+[1]*1000)
train_labels = np.array([])
for _ in range(10):
    train_labels = np.concatenate((train_labels,labels))

test_data = np.load(open('bottleneck_features_test.npy','rb'))
labels =np.array([0]*500+[1]*500)
test_labels = np.array([])
for _ in range(10):
    test_labels = np.concatenate((test_labels,labels))

train_labels = np_utils.to_categorical(train_labels,num_classes=2)
test_labels = np_utils.to_categorical(test_labels,num_classes=2)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

adam = Adam(lr= 1e-4)

model.compile(optimizer=adam,loss='categorical_crossentrop',metrics=['accuracy'])

model.fit(
    train_data,
    epochs=20,
    batch_size=batch_size,
    validation_data=(test_data,test_labels)
)

model.sample_weights('bottleneck_fc_model.h5')