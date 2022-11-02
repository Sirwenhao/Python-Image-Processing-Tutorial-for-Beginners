# 2022/10/29 author:WH

import numpy as np
from scipy import signal, misc, ndimage
from skimage import filters, feature, img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
from PIL import Image, ImageFilter
import matplotlib.pylab as pylab

def plot_image(image, title):
    pylab.imshow(image), pylab.title(title, size=10), pylab.axis('off')


# # 拉普拉斯算子
# ker_laplacian = [[0, -1, 0],[-1, 4, -1],[0, -1, 0]]
# im = rgb2gray(imread('Chapter05\Ch05images\dog.jpg'))
# im1 = np.clip(signal.convolve2d(im, ker_laplacian, mode='same'), 0, 1)
# pylab.gray()
# pylab.figure(figsize=(20, 10))
# pylab.subplot(121),plot_image(im, 'Original')
# pylab.subplot(122),plot_image(im1, 'Lappacian convolved')
# pylab.show()

# # 使用拉普拉斯滤波器锐化图像
# from skimage.filters import laplace

# im = rgb2gray(imread('Chapter05\Ch05images\dog.jpg'))
# im1 = np.clip(laplace(im) + im, 0, 1)
# pylab.figure(figsize=(20, 30))
# pylab.subplot(121), plot_image(im, 'Original iamge')
# pylab.subplot(122), plot_image(im1, 'sharpened image')
# pylab.tight_layout()
# pylab.show()

# # 利用SciPy的ndimage模块对灰度图像执行反锐化掩模操作
# def rgb2gray(im):
#     return np.clip(0.2989*im[...,0] + 0.5870*im[...,1] + 0.1140*im[...,2], 0, 1)

# im = rgb2gray(img_as_float(imread('Chapter05\Ch05images\dog.jpg')))
# im_blurred = ndimage.gaussian_filter(im, 5)
# im_detail = np.clip(im - im_blurred, 0, 1)
# pylab.gray()
# fig, axes =pylab.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 10))
# axes = axes.ravel()
# axes[0].set_title('Original image', size=5),axes[0].imshow(im)
# axes[1].set_title('Blurred image, sigma=5', size=5),axes[1].imshow(im_blurred)
# axes[2].set_title('Detail image', size=5),axes[2].imshow(im_detail)
# alpha = [1, 5, 10]
# for i in range(3):
#     im_sharp = np.clip(im + alpha[i]*im_detail, 0, 1)
#     axes[3+i].imshow(im_sharp), axes[3+i].set_title('Sharpened image,alpha='+str(alpha[i]), size=5)
# for ax in axes:
#     ax.axis('off')
# fig.tight_layout()
# pylab.show()


# # 使用sobel算子的边缘检测
# im = rgb2gray(imread('Chapter05\Ch05images\dog.jpg'))
# pylab.gray()
# pylab.figure(figsize=(20, 18))
# pylab.subplot(2, 2, 1)
# plot_image(im, 'Original')
# pylab.subplot(2,2,2)
# edges_x = filters.sobel_h(im)
# plot_image(edges_x, 'sobel_x')
# pylab.subplot(2,2,3)
# edges_y = filters.sobel_v(im)
# plot_image(edges_y, 'sobel_y')
# pylab.subplot(2,2,4)
# edges = filters.sobel(im)
# plot_image(edges, 'sobel')
# pylab.subplots_adjust(wspace=0.1, hspace=0.1)
# pylab.show()


# # scikit-image中的边缘检测器：Prewitt、Roberts、Sobel、Scharr和Laplace
# im = rgb2gray(imread('Chapter05\Ch05images\golden state bridge.jpg'))
# pylab.gray()
# pylab.figure(figsize=(15, 10))
# pylab.subplot(3,2,1),plot_image(im, 'Original')
# edges = filters.roberts(im)
# pylab.subplot(3,2,2),plot_image(edges, 'Roberts')
# edges = filters.scharr(im)
# pylab.subplot(3,2,3),plot_image(edges, 'Scharr')
# edges = filters.sobel(im)
# pylab.subplot(3,2,4),plot_image(edges, 'Sobel')
# edges = filters.prewitt(im)
# pylab.subplot(3,2,5),plot_image(edges, 'Prewitt')
# edges = np.clip(filters.laplace(im), 0, 1)
# pylab.subplot(3,2,6), plot_image(edges, 'Laplace')
# pylab.subplots_adjust(wspace=0.2, hspace=0.2)
# pylab.show()


# scikit-image中的canny边缘检测器
im = rgb2gray(imread('Chapter05\Ch05images\golden state bridge.jpg'))
im = ndimage.gaussian_filter(im, 4)
im += 0.05 * np.random.random(im.shape)
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)
fig, (axes1, axes2, axes3) = pylab.subplots(nrows=1, ncols=3, figsize=(20, 10), sharex=True, sharey=True)
axes1.imshow(im, cmap=pylab.cm.gray), axes1.axis('off'), axes1.set_title('noisy image', fontsize=10)
axes2.imshow(edges1, cmap=pylab.cm.gray), axes2.axis('off'), axes2.set_title('Canny filter, $\sigma=1$', fontsize=10)
axes3.imshow(edges2, cmap=pylab.cm.gray), axes3.axis('off'), axes3.set_title('Canny filter, $\sigma=3$', fontsize=10)
fig.tight_layout()
pylab.show()

