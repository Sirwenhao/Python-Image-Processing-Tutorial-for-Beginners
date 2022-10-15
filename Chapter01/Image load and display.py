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
im = imread('Chapter01\images\parrot.jpg')
im_hsv = color.rgb2hsv(im)
plt.gray()
plt.figure(figsize=(10,8))
plt.subplot(221), plt.imshow(im_hsv[...,0]),plt.title('h', size=20),plt.axis('off')
# plt.subplot(222), plt.imshow(im_hsv[...,1]),plt.title('s', size=20),plt.axis('off')
# plt.subplot(223), plt.imshow(im_hsv[...,2]),plt.title('v', size=20),plt.axis('off')
plt.subplot(224), plt.axis('off'), plt.show()