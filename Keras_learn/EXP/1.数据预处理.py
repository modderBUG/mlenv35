from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

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

# 载入图片
img = load_img('')
x = img_to_array(img)
x = x.shape((1,)+x.shape)

i = 0
for batch in datagen.flow(x,batch_size=1,save_to_dir='temp',save_prefix='cat',save_format='jpeg'):
    i+=1
    if i>20:
        break