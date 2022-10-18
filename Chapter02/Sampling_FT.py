# 2022/10/17  author:WH
# chapter02示例
from PIL import Image
from skimage.io import imread, imshow, show
import scipy.fftpack as fp
from scipy import ndimage, misc, signal
# from scipy.stats import signaltonoise # 信噪比函数在scipy版本1.0中被deprecate
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

# # Down-sampling
# im = Image.open('Chapter02\Ch02images\\tajmahal.jpg')
# # im.show()
# # 降采样
# im1 = im.resize((im.width//5, im.height//5))
# pylab.figure(figsize=(15, 10)), pylab.imshow(im1), pylab.title('Down-Sampling', size=10), pylab.show()
# # 降采样+抗混叠
# im2 = im.resize((im.width//5, im.height//5), Image.ANTIALIAS)
# pylab.figure(figsize=(15, 10)), pylab.imshow(im2), pylab.title('Down-Sampling+Antialia', size=10), pylab.show()

# 使用PIL库量化图像
# 实现SNR函数
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

# im = Image.open('Chapter01\Ch01images\parrot.jpg')
# pylab.figure(figsize=(20, 20))
# num_colors_list = [1 << n for n in range(8,0,-1)]
# snr_list = []
# i = 1
# for num_colors in num_colors_list:
#     im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
#     pylab.subplot(4,2,i), pylab.imshow(im1), pylab.axis('off')
#     snr_list.append(signaltonoise(im1, axis=None))
#     pylab.title('Image with # colors = ' + str(num_colors) + ' SNR = ' + str(np.round(snr_list[i-1], 3)), size=5)
#     i += 1
# pylab.subplots_adjust(wspace=0.2, hspace=0.2)
# pylab.show()

# pylab.plot(num_colors_list, snr_list, 'r.-')
# pylab.xlabel('Max# colors in the image')
# pylab.ylabel('SNR')
# pylab.title('Change in SNR w.r.t # colors')
# pylab.xscale('log', base=2) # basex has been renamed 'base'
# pylab.gca().invert_xaxis()
# pylab.show()


# 使用FFT计算DFT
# 使用scipy.fftpack模块
# im = np.array(Image.open('Chapter01\Ch01images\\rhino.jpg'))
# SNR = signaltonoise(im, axis=None)
# print('SNR for the original image = ' + str(SNR))
# freq = fp.fft2(im)
# im1 = fp.ifft2(freq).real
# SNR = signaltonoise(im1, axis=None)
# print('SNR for the image obtained after reconstruction = ' + str(SNR))
# assert(np.allclose(im, im1))
# pylab.figure(figsize=(20, 10))
# pylab.subplot(121),pylab.imshow(im, cmap='gray'),pylab.axis('off')
# pylab.title('Original Image', size=10)
# pylab.subplot(122),pylab.imshow(im1, cmap='gray'),pylab.axis('off')
# pylab.title('Image obtained after reconstruction', size=10)
# pylab.show()
# freq2 = fp.fftshift(freq)
# pylab.figure(figsize=(10,10)),pylab.imshow((20*np.log10(0.1+freq2)).astype(int)), pylab.show()

# numpy的FFT模块
import numpy.fft as fp
im1 = rgb2gray(imread('Chapter01\Ch01images\house.jpg'))
pylab.figure(figsize=(12, 10))
freq1 = fp.fft2(im1)
im1_ = fp.ifft2(freq1).real
pylab.subplot(2,2,1),pylab.imshow(im1, cmap='gray'),pylab.title('Original Image', size=10)
pylab.subplot(2,2,2),pylab.imshow(20*np.log10(0.01+np.abs(fp.fftshift(freq1))),cmap='gray')
pylab.title('FFT Spectrum Magntitude',size=10)
pylab.subplot(2,2,3),pylab.imshow(np.angle(fp.fftshift(freq1)),cmap='gray')
pylab.title('FFT Phase',size=10)
pylab.subplot(2,2,4),pylab.imshow(np.clip(im1_,0,255),cmap='gray')
pylab.title('Reconstructed Image', size=10)
pylab.show()

# import numpy.fft as fp
im2 = rgb2gray(imread('Chapter01\Ch01images\house2.jpg'))
pylab.figure(figsize=(12, 10))
freq2 = fp.fft2(im2)
im2_ = fp.ifft2(freq2).real
pylab.subplot(2,2,1),pylab.imshow(im2, cmap='gray'),pylab.title('Original Image', size=10)
pylab.subplot(2,2,2),pylab.imshow(20*np.log10(0.01+np.abs(fp.fftshift(freq2))),cmap='gray')
pylab.title('FFT Spectrum Magntitude',size=10)
pylab.subplot(2,2,3),pylab.imshow(np.angle(fp.fftshift(freq2)),cmap='gray')
pylab.title('FFT Phase',size=10)
pylab.subplot(2,2,4),pylab.imshow(np.clip(im2_,0,255),cmap='gray')
pylab.title('Reconstructed Image', size=10)
pylab.show()

pylab.figure(figsize=(20, 15))
im1_ = fp.ifft2(np.vectorize(complex)(freq1.real, freq2.imag)).real
im2_ = fp.ifft2(np.vectorize(complex)(freq2.real, freq1.imag)).real
pylab.subplot(221),pylab.imshow(np.clip(im1_,0,255),cmap='gray')
pylab.title('Reconstructed Image (Re(F1) + Im(F2))',size=10)
pylab.subplot(212).pylab.inshow(np.clip(im2_,0,255),cmap='gray')
pylab.title('Reconstructed Image (Re(F2) + Im(F1))', size=10)
pylab.show()