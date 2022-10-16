# 四种读取图像的方式

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PIL.ImageChops import add, subtract, multiply, difference,screen
import PIL.ImageStat as stat
from skimage.io import imread, imsave, imshow, show, imread_collection,imshow_collection
from skimage import color, viewer, exposure, img_as_float, data
from skimage.transform import SimilarityTransform, warp, swirl
from skimage.util import invert, random_noise, montage
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from scipy.ndimage import affine_transform, zoom
from scipy import misc

# # 使用PIL读取、保存和显示图像
# im = Image.open('Chapter01\images\parrot.jpeg')
# print(im.width, im.height, im.mode, im.format, type(im))
# # 5464 8192 RGB JPEG <class 'PIL.JpegImagePlugin.JpegImageFile'>
# im.show()
# im_g = im.convert('L')
# im_g.save('Chapter01\images\parrot_gray.png')
# Image.open('Chapter01\images\parrot_gray.png').show()


# # 使用matplotlib读取、保存和显示图像
# im = mpimg.imread('Chapter01\images\hill.jpg')
# print(im.shape, im.dtype, type(im))
# plt.figure(figsize=(8,8))
# plt.imshow(im)
# plt.axis('off')
# plt.show()
# im_1 = im
# im_1[im_1 < 0.5] = 0
# plt.imshow(im_1)
# plt.axis('off')
# plt.tight_layout()
# plt.savefig('Chapter01\images\hill_dark.jpg')
# im = mpimg.imread('Chapter01\images\hill_dark.jpg')
# plt.figure(figsize=(10, 10))
# plt.imshow(im)
# plt.axis('off')
# plt.tight_layout()
# plt.show()


# # 使用matplotlib imshow()在显示时插值
# im = mpimg.imread('Chapter01\images\Lenna.jpg')
# methods = ['none', 'nearest','bilinear','bicubic','spline16','lanczos']
# fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(8,16),subplot_kw={'xticks':[],'yticks':[]})
# fig.subplots_adjust(hspace=0.1,wspace=0.1)
# for ax, interp_method in zip(axes.flat, methods):
#     ax.imshow(im, interpolation=interp_method)
#     ax.set_title(str(interp_method), size=10)
# plt.tight_layout()
# plt.show()

# 使用scikit-image读取、保存和显示图像
# im = imread('Chapter01\images\parrot.jpeg')
# print(im.shape, im.dtype, type(im))
# hsv = color.rgb2hsv(im)
# hsv[:,:,1] = 0.5
# im1 = color.hsv2rgb(hsv)
# imsave('Chapter01\images\parrot_hsv.png', im1)
# im=imread('Chapter01\images\parrot_hsv.png')
# plt.axis('off'), imshow(im), show()
# viewer = viewer.ImageViewer(im)
# viewer.show()

# im = data.astronaut()
# imshow(im), show()

# 使用SciPy的misc模块读取、保存和显示图像
# im = misc.face()
# imsave('Chapter01\images\\face.png', im)
# plt.imshow(im), plt.axis('off'), plt.show()

# 使用misc.imread从磁盘加载图像
# im = imread('Chapter01\images\pepper.jpg')
# print(type(im), im.shape, im.dtype)

# 使用imageio.imread()，并使用matplotlib显示图像
# import imageio
# im = imageio.imread('Chapter01\images\pepper.jpg')
# print(type(im), im.shape, im.dtype)
# plt.imshow(im), plt.axis('off'), plt.show()

# 将图像从RGB空间转换到HSV空间
# im = imread('Chapter01\images\parrot.jpg')
# im_hsv = color.rgb2hsv(im)
# plt.gray()
# plt.figure(figsize=(10,8))
# plt.subplot(221), plt.imshow(im_hsv[...,0]),plt.title('h', size=20),plt.axis('off')
# plt.subplot(222), plt.imshow(im_hsv[...,1]),plt.title('s', size=20),plt.axis('off')
# plt.subplot(223), plt.imshow(im_hsv[...,2]),plt.title('v', size=20),plt.axis('off')
# plt.subplot(224), plt.axis('off'), plt.show()

# 转换图像的数据结构,从PIL的Image对象转换为numpy的ndarray结构
# im = Image.open('Chapter01\images\\flower.jpg')
# im = np.array(im)
# imshow(im)
# plt.axis('off'), show()
# # 从ndarray转换为Image结构
# im = imread('Chapter01\images\\flower.jpg')
# im = Image.fromarray(im)
# im.show()

# 使用numpy数组的切片进行图像处理
# lena = mpimg.imread('Chapter01\images\Lenna.jpg')
# # print(lena[0, 40])
# # print(lena[10:13, 20:23, 0:1])
# # print(lena.shape) # (316, 316, 3)
# lx, ly, _ = lena.shape
# x, y = np.ogrid[0:lx, 0:ly]
# mask = (x - lx/2)**2 + (y - ly/2)**2 > lx * ly / 4
# lena[mask, :] = 0
# plt.figure(figsize=(10, 10))
# plt.imshow(lena), plt.axis('off'), plt.show()

# 使用交叉溶解的两个图像的α混合
# im1 = mpimg.imread('Chapter01\images\Lenna.jpg') / 255
# im2 = mpimg.imread('Chapter01\images\parrot.jpg') /255
# im2.resize(316, 316, 3)
# i = 1
# plt.figure(figsize=(18, 15))
# for alpha in np.linspace(0, 1, 20):
#     plt.subplot(4,5,i)
#     plt.imshow((1-alpha)*im1 + alpha*im2)
#     plt.axis('off')
#     i += 1
# plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.show()

# 使用PIL进行图像处理
# im = Image.open('Chapter01\images\parrot.jpg')
# print(im.width, im.height, im.mode, im.format)

# 使用PIL执行不同类型的图像操作

# 裁剪图像
# im_c = im.crop((1750,750,3200,2000))
# im_c.show()

# 调整图像的尺寸
# im = Image.open('Chapter01\images\clock.jpg')
# print(im.width, im.height)
# im.show()
# im_large = im.resize((im.width*5, im.height*5), Image.BILINEAR)
# im_large.show()
# im_small = im.resize((im.width//5, im.height//5), Image.ANTIALIAS)
# print(im_small.width, im_small.height)
# im_small.show()

# 图像负片
im = Image.open('Chapter01\images\parrot.jpg')
# im_t = im.point(lambda x: 255-x)
# im_t.show()
# 转换为灰度图像
# im_g = im.convert('L')
# im_g.show()

# 灰度级变换
# im_g.point(lambda x: 255*np.log(1+x/255)).show()
# 幂律变换
# im_g.point(lambda x: 255*(x/255)**0.6).show()

# 计算图像的基本统计信息
# s = stat.Stat(im)
# print(s.extrema)
# print(s.count)
# print(s.mean)
# print(s.median)
# print(s.stddev)

# 绘制图像的RGB通道的像素值的直方图，histogram()函数用于统计每个通道像素的直方图（像素值与频率表）
# pl = im.histogram()
# plt.bar(range(256), pl[:256], color='r', alpha=0.5)
# plt.bar(range(256), pl[256:2*256], color='g', alpha=0.4)
# plt.bar(range(256), pl[256*2:], color='b', alpha=0.3)
# plt.show()

# 分离图像的RGB三个颜色通道,split()函数
ch_r, ch_g, ch_b = im.split()
plt.figure(figsize=(18, 6))
plt.subplot(1,3,1), plt.imshow(ch_r, cmap=plt.cm.Reds), plt.axis('off')
plt.subplot(1,3,2), plt.imshow(ch_g, cmap=plt.cm.Greens), plt.axis('off')
plt.subplot(1,3,3), plt.imshow(ch_b, cmap=plt.cm.Blues), plt.axis('off')
plt.tight_layout()
plt.show()
