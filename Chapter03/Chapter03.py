# 卷积和滤波器操作

# 使用numpy的fft模块实现高斯模糊滤波器
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

# # 信噪比函数需要自己写
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd) 
    # np.where(condition,x,y)第一个参数表示条件，条件成立返回x，不成立返回y

# pylab.figure(figsize=(20,15))
# pylab.gray() # show the filtered result in grayscale
# im = np.mean(imread('Chapter01\Ch01images\Lenna.jpg'), axis=2)
# gauss_kernel = np.outer(signal.gaussian(im.shape[0], 5), signal.gaussian(im.shape[1], 5))
# freq = fp.fft2(im)
# assert(freq.shape == gauss_kernel.shape)
# freq_kernel = fp.fft2(fp.ifftshift(gauss_kernel))
# convolved = freq*freq_kernel # by the convolution theorem, simply multiply in the frequency domain
# im1 = fp.ifft2(convolved).real
# pylab.subplot(2,3,1), pylab.imshow(im), pylab.title('Original Image', size=10), pylab.axis('off')
# pylab.subplot(2,3,2), pylab.imshow(gauss_kernel), pylab.title('Gaussian Kernel', size=10)
# pylab.subplot(2,3,3), pylab.imshow(im1) # the imaginary part is an artifact
# pylab.title('Output Image', size=10), pylab.axis('off')
# pylab.subplot(2,3,4), pylab.imshow( (20*np.log10( 0.1 + fp.fftshift(freq))).astype(int))
# pylab.title('Original Image Spectrum', size=10), pylab.axis('off')
# pylab.subplot(2,3,5), pylab.imshow( (20*np.log10( 0.1 + fp.fftshift(freq_kernel))).astype(int))
# pylab.title('Gaussian Kernel Spectrum', size=10), pylab.subplot(2,3,6)
# pylab.imshow( (20*np.log10( 0.1 + fp.fftshift(convolved))).astype(int))
# pylab.title('Output Image Spectrum', size=10), pylab.axis('off')
# pylab.subplots_adjust(wspace=0.2, hspace=0.2)
# pylab.show()

# # Gaussian kernel in the frequency domain
# im = rgb2gray(imread('Chapter01\Ch01images\Lenna.jpg'))
# gauss_kernel = np.outer(signal.gaussian(im.shape[0], 1), signal.gaussian(im.shape[1], 1))
# freq = fp.fft2(im)
# freq_kernel = fp.fft2(fp.ifftshift(gauss_kernel))
# pylab.imshow((20*np.log10(0.01+fp.fftshift(freq_kernel))).real.astype(int), cmap='coolwarm')
# pylab.colorbar()
# pylab.show()

# High Pass Filter (HPF)
# im = np.array(Image.open('Chapter01\Ch01images\\rhino.jpg').convert('L'))
# pylab.figure(figsize=(10, 10)), pylab.imshow(im, cmap=pylab.cm.gray), pylab.axis('off'), pylab.show()
# freq = fp.fft2(im)
# (w, h) = freq.shape
# half_w, half_h = int(w/2), int(h/2)
# freq1 = np.copy(freq)
# freq2 = fp.fftshift(freq1)
# pylab.figure(figsize=(10, 10)), pylab.imshow((20*np.log10(0.1+freq2)).astype(int)), pylab.show()

# HPF用于灰度图像
# from scipy import fftpack
# im = np.array(Image.open('Chapter01\Ch01images\cameraman.jpg').convert('L'))
# freq = fp.fft2(im)
# # print(freq.shape)
# (w, h) = freq.shape
# half_w, half_h = int(w/2), int(h/2)
# snrs_hp = []
# lbs = list(range(1, 25))
# pylab.figure(figsize=(12, 20))
# for l in lbs:
#     freq1 = np.copy(freq)
#     freq2 = fftpack.fftshift(freq1)
#     freq2[half_w-1:half_w+1+1, half_h-1:half_h+1+1] = 0
#     im1 = np.clip(fp.ifft2(fftpack.ifftshift(freq2)).real, 0, 255)
#     snrs_hp.append(signaltonoise(im1, axis=None))
#     pylab.subplot(6,4,l), pylab.imshow(im1, cmap='gray'), pylab.axis('off')
#     pylab.title('F = ' + str(l+1), size=8)
# pylab.subplots_adjust(wspace=0.1, hspace=0.3)
# pylab.show()
# pylab.plot(lbs, snrs_hp, 'b.-')
# pylab.xlabel('Cutoff Frequency for HPF', size=8)
# pylab.ylabel('SNR', size=8)
# pylab.show()

