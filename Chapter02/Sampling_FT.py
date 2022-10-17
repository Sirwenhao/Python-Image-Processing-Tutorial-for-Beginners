# 2022/10/17  author:WH
# chapter02示例
from PIL import Image
from skimage.io import imread, imshow, show
import scipy.fftpack as fp
from scipy import ndimage, misc, signal
# from scipy.stats import signaltonoise
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.transform import rescale
import matplotlib.pylab as pylab
import numpy as np
import numpy.fft
import timeit

# # Up-Sampling
# im = Image.open('Chapter02\Ch02images\clock.jpg')
# pylab.imshow(im), pylab.title('Original', size=10), pylab.show()
# # 上采样(nearest neighbor interpolation)
# im1 = im.resize((im.width*5, im.height*5), Image.NEAREST)
# pylab.figure(figsize=(10, 10)), pylab.imshow(im1), pylab.title('NEAREST', size=10), pylab.show()
# # 上采样(bi-linear interpolation)
# im2 = im.resize((im.width*5, im.height*5), Image.BILINEAR)
# pylab.figure(figsize=(10, 10)), pylab.imshow(im2), pylab.title('BILINEAR', size=10), pylab.show()
# # 上采样(bi-cubic interpolation)
# im3 = im.resize((im.width*10, im.height*10), Image.BICUBIC)
# pylab.figure(figsize=(10, 10)), pylab.imshow(im3), pylab.title('BICUBIC', size=10), pylab.show()

# Down-sampling
im = Image.open('Chapter02\Ch02images\\tajmahal.jpg')
# im.show()
# 降采样
im1 = im.resize((im.width//5, im.height//5))
pylab.figure(figsize=(15, 10)), pylab.imshow(im1), pylab.title('Down-Sampling', size=10), pylab.show()
# 降采样+抗混叠
im2 = im.resize((im.width//5, im.height//5), Image.ANTIALIAS)
pylab.figure(figsize=(15, 10)), pylab.imshow(im2), pylab.title('Down-Sampling+Antialia', size=10), pylab.show()


