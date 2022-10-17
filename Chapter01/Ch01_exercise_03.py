# 创建Python版本的Gotham Instagram滤波器
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

im = Image.open('Chapter01\Ch01images\\flower.jpg')
print(np.max(im))
plt.figure(figsize=(200, 200))
plt.imshow(im)
plt.axis('off')
plt.show()

r,g,b = im.split()
r_old = np.linspace(0,255,11) # np.linspace()创建等差数列
r_new = [0., 12.75, 25.5, 51., 76.5, 127.5, 178.5, 204., 229.5, 242.25, 255.] # 所用到的11个插值
# strech the red channel histogram with interpolation and obtain new red channel values for each pixel 
r1 = Image.fromarray((np.reshape(np.interp(np.array(r).ravel(), r_old, r_new),
                                 (im.height, im.width))).astype(np.uint8), mode='L')

plt.figure(figsize=(200,150))
plt.subplot(221) # plt.subplot(nrows, ncols, index, **kwargs)=plt.subplot(nrows ncols index)(使用时不加空格，这里为了区分)
# plt.subplot(221)相当于plt.subplot(2,2,1)数字代表绘制图像的位置，第一个数字代表行数，第二个数字代表列数，第三个数字表示索引
plt.imshow(im)
plt.title('original', size=20)
plt.axis('off')
plt.subplot(222)
im1 = Image.merge('RGB', (r1, r1, b))
plt.imshow(im1)
plt.axis('off')
plt.title('with red channel interpolation', size=20)
plt.subplot(223)
plt.hist(np.array(r).ravel(), density=True, stacked=True)
plt.subplot(224)
plt.hist(np.array(r1).ravel(), density=True, stacked=True)
plt.show()

plt.figure(figsize=(200, 100))
plt.subplot(121)
plt.imshow(im1)
plt.title('last image', size=20)
plt.axis('off')
b1 = Image.fromarray(np.clip(np.array(b) + 7.65, 0, 255).astype(np.uint8))
im1 = Image.merge('RGB', (r1, g, b1))
plt.subplot(122)
plt.imshow(im1)
plt.axis('off')
plt.title('with transformation', size=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(200,200))
plt.imshow(im1)
plt.axis('off')
plt.title('Final Image', size=20)
plt.tight_layout()
plt.show()