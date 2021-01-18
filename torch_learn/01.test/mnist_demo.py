from torchvision import transforms
import numpy as np

data = np.random.randint(0, 255, size=12)
img = data.reshape(2, 2, 3)
print(img.shape)

img_tensor = transforms.ToTensor()(img)  # 转换成tensor
print(img_tensor)
print(img_tensor.shape)

import os

os.makedirs()