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

# # # 实现输入图像的颜色通道直方图
def plot_image(image, title=''):
    pylab.title(title, size=10), pylab.imshow(image)
    pylab.axis('off')

# def plot_hist(r, g, b, title=''):
#     r, g, b = img_as_ubyte(r), img_as_ubyte(g), img_as_ubyte(b)
#     pylab.hist(np.array(r).ravel(), bins=256, range=(0, 256), color='r', alpha=0.5)
#     pylab.hist(np.array(g).ravel(), bins=256, range=(0, 256), color='g', alpha=0.5)
#     pylab.hist(np.array(b).ravel(), bins=256, range=(0, 256), color='b', alpha=0.5)
#     pylab.xlabel('pixel value', size=10), pylab.ylabel('frequency', size=10)
#     pylab.title(title, size=10)

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


# # 幂律变换
# im = img_as_float(imread('Chapter01\Ch01images\earthfromsky.jpg'))
# gamma = 5
# im1 = im**gamma
# pylab.style.use('ggplot')
# pylab.figure(figsize=(15, 5))
# pylab.subplot(121), plot_hist(im[..., 0], im[..., 1], im[..., 2], 'histogram for RGB channels (iutput)')
# pylab.subplot(122), plot_hist(im1[..., 0], im1[..., 1], im1[..., 2], 'histogram for RGB channels (output)')
# pylab.show()

# # # PIL作为点操作
# im = Image.open('Chapter01\Ch01images\cheetah.jpg')
# im_r, im_g, im_b = im.split()
# pylab.style.use('ggplot')
# pylab.figure(figsize=(15, 5))
# pylab.subplot(121)
# plot_image(im)
# pylab.subplot(122)
# plot_hist(im_r, im_g, im_b)
# pylab.show()


# # 使用PIL的point()函数实现对比度拉伸
# def contrast(c):
#     return 0 if c < 70 else (255 if c > 150 else (255*c-22950)/48)

# im1 = im.point(contrast)
# im_r, im_g, im_b = im1.split()
# pylab.style.use('ggplot')
# pylab.figure(figsize=(15, 5))
# pylab.subplot(121)
# plot_image(im1)
# pylab.subplot(122)
# plot_hist(im_r, im_g, im_b)
# pylab.yscale('log',base=10)
# pylab.show()

# # PIL的ImageEnhance模块
# contrast = ImageEnhance.Contrast(im)
# im1 = np.reshape(np.array(contrast.enhance(2).getdata()).astype(np.uint8), (im.height, im.width, 3))
# pylab.style.use('ggplot')
# pylab.figure(figsize=(15, 5))
# pylab.subplot(121), plot_image(im1)
# pylab.subplot(122), plot_hist(im1[...,0], im1[...,1], im1[..., 2]), pylab.yscale('log', base=10)
# pylab.show()


# # 固定阈值二值化，使用PIL的point()函数
# im = Image.open('Chapter04\Ch04images\swans.jpg').convert('L')
# pylab.hist(np.array(im).ravel(),bins=256,range=(0, 256),color='g')
# pylab.xlabel('Pixel values'), pylab.ylabel('Frequency'), pylab.title('Histogram f pixel values')
# pylab.show()
# pylab.figure(figsize=(12, 18))
# pylab.gray()
# pylab.subplot(221),plot_image(im, 'Orginal image'), pylab.axis('off')
# th = [0, 50, 100, 150, 200]
# for i in range(2, 5):
#     im1 = im.point(lambda x: x > th[i])
#     pylab.subplot(2,2,i), plot_image(im1, 'binary image with threshold = ' + str(th[i]))
# pylab.show()

# # 半色调二值化
# im = Image.open('Chapter04\Ch04images\swans.jpg').convert('L')
# im = Image.fromarray(np.clip(im + np.random.randint(-128, 128,(im.height, im.width)), 0, 255).astype(np.uint8))
# pylab.figure(figsize=(12, 18))
# pylab.subplot(221),plot_image(im, 'Original image (with noise)')
# th = [0, 50, 100, 150, 200]
# for i in range(2, 5):
#     im1 = im.point(lambda x: x > th[i])
#     pylab.subplot(2,2,i), plot_image(im1, 'binary image with threshold='+ str(th[i]))
# pylab.show()

# 基于误差扩散的Floyd-Steinberg抖动
# import cv2
# import matplotlib.pyplot as plt

# img_gray = cv2.imread('Chapter04\Ch04images\swans.jpg', cv2.IMREAD_GRAYSCALE)
# img_gray0 = 255 - img_gray
# h, w = img_gray0.shape
# img_gray0 =cv2.resize(img_gray0, (w//2, h//2))
# h,w =img_gray0.shape
# plt.figure()
# plt.imshow(img_gray0, vmin=0, vmax=255, cmap=plt.get_cmap("Greys"))
# plt.title("Original Image")
# img_gray_eq = img_gray0
# img_dither =np.zeros((h+1, w+1), dtype=np.float)
# img_undither = np.zeros((h, w), dtype=np.uint8)

# threshold = 128

# for i in range(h):
#     for j in range(w):
#         img_dither[i, j] = img_gray_eq[i, j]
#         if img_gray_eq[i, j] > threshold:
#             img_undither[i, j] = 255

# for i in range(h):
#     for j in range(w):
#         old_pix = img_dither[i, j]
#         if (img_dither[i, j] > threshold):
#             new_pix = 255
#         else:
#             new_pix = 0

#         img_dither[i, j] = new_pix
#         quant_err = old_pix - new_pix
#         if j > 0:
#             img_dither[i+1, j-1] = img_dither[i+1, j-1] + quant_err * 3 / 16
#             img_dither[i+1, j] = img_dither[i+1, j] + quant_err * 5 / 16
#             img_dither[i, j+1] = img_dither[i, j+1] + quant_err * 7 / 16
#             img_dither[i+1, j+1] = img_dither[i+1, j+1] + quant_err * 1 / 16

# img_dither = img_dither.astype(np.uint8)
# img_dither = img_dither[0:h, 0:w]

# plt.figure()
# plt.imshow(img_dither, vmin=0, vmax=255, cmap=plt.get_cmap("Greys"))
# plt.title("dither")

# plt.figure()
# plt.imshow(img_undither, vmin=0, vmax=255, cmap=plt.get_cmap("Greys"))
# plt.title("undither")

# plt.show()



# # 基于scikit-image的对比度拉伸和直方图均衡化
# img = rgb2gray(imread('Chapter01\Ch01images\earthfromsky.jpg'))
# # histogram equalization
# img_eq = exposure.equalize_hist(img)
# # adaptive histogram equalization
# img_adapteq = exposure.equalize_adapthist(img , clip_limit=0.03)
# pylab.gray()
# images = [img, img_eq, img_adapteq]
# titles = ['Original input (earth from sky)', 'after histogram equalization', 'after adaptive histogram equalization']
# for i in range(3):
#     pylab.figure(figsize=(20, 10)), plot_image(images[i], titles[i])
# pylab.figure(figsize=(15, 5))
# for i in range(3):
#     pylab.subplot(1, 3, i+1), pylab.hist(images[i].ravel(), color='g'),pylab.title(titles[i], size=10)
# pylab.show()


