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


# # scikit-image中的canny边缘检测器
# im = rgb2gray(imread('Chapter05\Ch05images\golden state bridge.jpg'))
# im = ndimage.gaussian_filter(im, 4)
# im += 0.05 * np.random.random(im.shape)
# edges1 = feature.canny(im)
# edges2 = feature.canny(im, sigma=3)
# fig, (axes1, axes2, axes3) = pylab.subplots(nrows=1, ncols=3, figsize=(20, 10), sharex=True, sharey=True)
# axes1.imshow(im, cmap=pylab.cm.gray), axes1.axis('off'), axes1.set_title('noisy image', fontsize=10)
# axes2.imshow(edges1, cmap=pylab.cm.gray), axes2.axis('off'), axes2.set_title('Canny filter, $\sigma=1$', fontsize=10)
# axes3.imshow(edges2, cmap=pylab.cm.gray), axes3.axis('off'), axes3.set_title('Canny filter, $\sigma=3$', fontsize=10)
# fig.tight_layout()
# pylab.show()


# # LoG滤波器和DoG滤波器
# from scipy.signal import convolve2d
# from scipy.ndimage import gaussian_filter
# from numpy import pi

# def plot_kernel(kernel, s, name):
#     pylab.imshow(kernel, cmap='YlOrRd')

# # 定义LoG滤波器
# def LoG(k=12, s=3):
#     n = 2*k+1 # size of kernel
#     kernel = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             kernel[i, j] = -(1-((i-k)**2+(j-k)**2)/(2.*s**2))*np.exp(-((i-k)**2+(j-k)**2)/(2.*s**2))/(pi*s**4)
#     kernel = np.round(kernel / np.sqrt((kernel**2).sum()),3)
#     return kernel

# # 定义DoG滤波核
# def DoG(k=12, s=3):
#     n = 2*k+1 # size of the kernel
#     s1, s2 = s * np.sqrt(2), s / np.sqrt(2)
#     kernel = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             kernel[i, j] = np.exp(-((i-k)**2+(j-k)**2)/(2.*s1**2))/(2*pi*s1**2)
#             - np.exp(-((i-k)**2+(j-k)**2)/(2.*s2**2))/(2*pi*s2**2)
#     kernel = np.round(kernel/np.sqrt((kernel**2).sum()), 3)
#     return kernel

# s = 3 # sigma value of LoG
# im = rgb2gray(imread('Chapter05\Ch05images\dog.jpg'))
# kernel = LoG()
# outimg1 = convolve2d(im, kernel)
# pylab.subplot(221), pylab.title('LoG kernel', size=10), plot_kernel(kernel, s, 'LoG')
# pylab.subplot(222), pylab.title('Output image with LoG', size=10)
# pylab.imshow(np.clip(outimg1, 0, 1), cmap='gray') # clip the pixel values in between 0 and 1
# kernel = DoG()
# outimg2 = convolve2d(im, kernel)
# pylab.subplot(223), pylab.title('DoG kernel', size=10), plot_kernel(kernel, s, 'DoG')
# pylab.subplot(224), pylab.title('Output image with DoG', size=10)
# pylab.imshow(np.clip(outimg2, 0, 1), cmap='gray')
# pylab.tight_layout()
# pylab.show()

# # 2022/11/24 author:WH
# # scikit-image transform pyramid模块中的高斯金字塔
# from skimage.transform import pyramid_gaussian
# image = imread('Chapter01\Ch01images\Lenna.jpg')
# nrows, ncols = image.shape[:2]
# pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True))
# pylab.figure(figsize=(20, 5))
# i, n = 1, len(pyramid)
# for p in pyramid:
#     pylab.subplot(1, n, i), pylab.imshow(p)
#     pylab.title(str(p.shape[0]) + 'x' + str(p.shape[1])), pylab.axis('off')
#     i += 1
# pylab.suptitle('Gaussian Pyramid', size=6)
# pylab.show()
# compos_image = np.zeros((nrows, ncols+ncols//2, 3), dtype=np.double)
# compos_image[:nrows, :ncols, :] = pyramid[0]
# i_row = 0
# for p in pyramid[1:]:
#     nrows, ncols = p.shape[:2]
#     compos_image[i_row:i_row+nrows, ncols:ncols+ncols] = p
#     i_row += nrows
# fig, axes = pylab.subplots(figsize=(20, 20))
# axes.imshow(compos_image)
# pylab.show()

# # scikit-image transform pyramid模块中的拉普拉斯金字塔
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.transform import pyramid_laplacian
# from skimage.color import rgb2gray
# image = imread('Chapter01\Ch01images\Lenna.jpg')
# nrows, ncols = image.shape[:2]
# pyramid = tuple(pyramid_laplacian(image, downscale=2, multichannel=True))
# plt.figure(figsize=(20, 20))
# i, n = 1, len(pyramid)
# for p in pyramid[:-1]: # tuple中全为0的那一组显示不了
#     plt.subplot(3,3,i), plt.imshow(rgb2gray(p), cmap='gray')
#     plt.title(str(p.shape[0]) + 'x' + str(p.shape[1]))
#     plt.axis('off')
#     i += 1
# plt.suptitle('Laplacian Pyramid', size=6)
# plt.show()
# composite_image = np.zeros((nrows, ncols + ncols // 2), dtype=np.double)
# composite_image[:nrows, :ncols] = rgb2gray(pyramid[0])
# i_row = 0
# for p in pyramid[1:]:
#     n_rows, n_cols = p.shape[:2]
#     composite_image[i_row:i_row + n_rows, ncols:ncols + n_cols] = rgb2gray(p)
#     i_row += n_rows
# fig, ax = plt.subplots(figsize=(20,20))
# ax.imshow(composite_image, cmap='gray')
# plt.show()


# # 2022/11/26 author:WH
# # Chapter05 Exercises

# # blending images with Pyramids

# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.io import imread
# from skimage.color import rgb2gray
# from skimage.transform import pyramid_reduce, pyramid_laplacian, pyramid_expand, resize

# image = imread('Chapter06\CH06images\\antelope.jpg')
# print(image.shape)

# def get_gaussian_pyramid(image):
#     rows, cols, dim = image.shape
#     gaussian_pyramid = [image]
#     while rows > 1 and cols > 1:
#         image = pyramid_reduce(image, downscale=2, channel_axis=3)
#         gaussian_pyramid.append(image)
#         rows //= 2
#         cols //= 2
#     return gaussian_pyramid

# def get_laplacian_pyramid(gaussian_pyramid):
#     laplacian_pyramid = [gaussian_pyramid[len(gaussian_pyramid)-1]]
#     for i in range(len(gaussian_pyramid)-2, -1, -1):
#         image = gaussian_pyramid[i] - resize(pyramid_expand(gaussian_pyramid[i+1]), gaussian_pyramid[i].shape)
#         laplacian_pyramid.append(np.copy(image))
#     laplacian_pyramid = laplacian_pyramid[::-1]
#     return laplacian_pyramid

# gaussian_pyramid = get_gaussian_pyramid(image)
# laplacian_pyramid = get_laplacian_pyramid(image)

# w, h = 20, 10
# for i in range(3):
#     plt.figure(figsize=(w, h))
#     p = gaussian_pyramid[i]
#     plt.imshow(p)
#     plt.title(str(p.shape[0])+'x'+str(p.shape[1]), size=8)
#     plt.axis('off')
#     w, h = w/2, h/2
#     plt.show()

# w, h = 10, 5
# for i in range(1, 4):
#     plt.figure(figsize=(w, h))
#     p = laplacian_pyramid[i]
#     plt.imshow(rgb2gray(p), cmap='gray')
#     plt.title(str(p.shape[0])+'x'+str(p.shape[1]), size=8)
#     plt.axis('off')
#     w, h = w/2, h/2
#     plt.show()

# Marr and Hildreth's zero-crossing algorithm for edge detection

import numpy as np
from scipy import ndimage,misc
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def any_neighbor_zero(img, i, j):
    for k in range(-1, 2):
        for l in range(-1, 2):
            if img[i+k, j+k] == 0:
                return True
    return False

def zero_crossing(img):
    img[img>0] = 1
    img[img<0] = 0
    out_img = np.zeros(img.shape)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] > 0 and any_neighbor_zero(img, i, j):
                out_img[i, j] = 255
    return out_img

img = rgb2gray(imread('Chapter06\CH06images\zebra.jpg'))
print(np.max(img))
plt.figure(figsize=(20,10))
plt.imshow(img)
plt.axis('off')
plt.title('Original Image', size=8)

fig = plt.figure(figsize=(25,15))
plt.gray() # show the filtered result in grayscale
for sigma in range(2, 10, 2):
    plt.subplot(2,2,int(sigma/2))
    result = ndimage.gaussian_laplace(img, sigma=sigma)
    result = zero_crossing(result)
    plt.imshow(result)
    plt.axis('off')
    plt.title('LoG with zero-crossing, sigma=' + str(sigma), size=8)

plt.tight_layout()
plt.show()
