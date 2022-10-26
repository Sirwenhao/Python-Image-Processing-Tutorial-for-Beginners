import numpy as np
from skimage import data, img_as_float, img_as_ubyte,exposure, io, color
from skimage.io import imread
from skimage.exposure import cumulative_distribution
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.util import random_noise
from skimage.color import rgb2gray
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage, misc
import matplotlib.pylab as pylab

# # 实现输入图像的颜色通道直方图
def plot_image(image, title=''):
    pylab.title(title, size=10), pylab.imshow(image)
    pylab.axis('off')

def plot_hist(r, g, b, title=''):
    r, g, b = img_as_ubyte(r), img_as_ubyte(g), img_as_ubyte(b)
    pylab.hist(np.array(r).ravel(), bins=256, range=(0, 256), color='r', alpha=0.5)
    pylab.hist(np.array(g).ravel(), bins=256, range=(0, 256), color='g', alpha=0.5)
    pylab.hist(np.array(b).ravel(), bins=256, range=(0, 256), color='b', alpha=0.5)
    pylab.xlabel('pixel value', size=10), pylab.ylabel('frequency', size=10)
    pylab.title(title, size=10)

# im = Image.open('Chapter01\Ch01images\parrot.jpg')
# im_r, im_g, im_b = im.split()
# pylab.style.use('ggplot')
# pylab.figure(figsize=(15, 5))
# pylab.subplot(121), plot_image(im, 'original image')
# pylab.subplot(122), plot_hist(im_r, im_g, im_b, 'historam for RGB channels')
# pylab.show()


# # 使用PIL的point()函数进行对数变换
# im = im.point(lambda i: 255*np.log(1+i/255))
# im_r, im_g, im_b = im.split()
# pylab.style.use('ggplot')
# pylab.figure(figsize=(15, 5))
# pylab.subplot(121), plot_image(im, 'image after log transform')
# pylab.subplot(122), plot_hist(im_r, im_g, im_b, 'histogram of RGB channels log transform')
# pylab.show()


# 幂律变换
im = img_as_float(imread('Chapter01\Ch01images\earthfromsky.jpg'))
gamma = 5
im1 = im**gamma
pylab.style.use('ggplot')
pylab.figure(figsize=(15, 5))
pylab.subplot(121), plot_hist(im[..., 0], im[..., 1], im[..., 2], 'histogram for RGB channels (iutput)')
pylab.subplot(122), plot_hist(im1[..., 0], im1[..., 1], im1[..., 2], 'histogram for RGB channels (output)')
pylab.show()

# PIL作为点操作
im = Image.open('Chapter01\Ch01images\cheetah.jpg')
im_r, im_g, im_b = im.split()
pylab.style.use('ggplot')
pylab.figure(figsize=(15, 5))
pylab.subplot(121)
plot_image(im)
pylab.subplot(122)
plot_hist(im_r, im_g, im_b)
pylab.show()

