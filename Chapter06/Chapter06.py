# 2022/11/26  author:WH
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pylab as pylab
from skimage.morphology import binary_erosion, rectangle

# 腐蚀
def plot_image(image, title=''):
    pylab.title(title, size=6), pylab.imshow(image)
    pylab.axis('off')

# im = rgb2gray(imread('Chapter06\CH06images\clock2.jpg'))
# im[im <= 0.5] = 0 # create binary image with fixed threshold=0.5
# im[im > 0.5] = 1
# pylab.gray()
# # pylab.figure(figsize=(20, 10))
# pylab.subplot(1, 3, 1), plot_image(im, 'Original')
# im1 = binary_erosion(im, rectangle(1, 5))
# pylab.subplot(1, 3, 2), plot_image(im1, 'Erosion with rectangle size (1, 5)')
# im1 = binary_erosion(im, rectangle(1, 15))
# pylab.subplot(1, 3, 3), plot_image(im1, 'Erosion with rectangle size (1, 15)')
# pylab.show()

# # 膨胀
# from skimage.morphology import binary_dilation, disk
from skimage import img_as_float

# im = img_as_float(imread('Chapter06\CH06images\\tagore.png'))
# im = 1 - im[...,3]
# im[im <= 0.5] = 0
# im[im > 0.5] = 1
# pylab.gray()
# pylab.subplot(131)
# pylab.imshow(im)
# pylab.title('Original', size=8)
# pylab.axis('off')
# for d in range(1, 3):
#     pylab.subplot(1, 3, d+1)
#     im1 = binary_dilation(im, disk(2*d))
#     pylab.imshow(im1)
#     pylab.title('dilation with disk size ' + str(2*d), size=8)
#     pylab.axis('off')
# pylab.show()


# # 开闭操作
from skimage.morphology import binary_opening, binary_closing, binary_erosion, binary_dilation, disk
# im = rgb2gray(imread('Chapter06\CH06images\circles.jpg'))
# im[im <= 0.5] = 0
# im[im > 0.5] = 1
# pylab.gray()
# # pylab.figure(figsize=(20,10))
# pylab.subplot(1,3,1), plot_image(im, 'original')
# im1 = binary_opening(im, disk(12))
# pylab.subplot(1,3,2), plot_image(im1, 'opening with disk size ' + str(12))
# im1 = binary_closing(im, disk(6))
# pylab.subplot(1,3,3), plot_image(im1, 'closing with disk size ' + str(6))
# pylab.show()

def plot_images_horizontally(original, filtered, filter_name, sz=(18,7)):
    pylab.gray()
    # pylab.figure(figsize = sz)
    pylab.subplot(1,2,1), plot_image(original, 'original')
    pylab.subplot(1,2,2), plot_image(filtered, filter_name)
    pylab.show()

# # 骨架化

# from skimage.morphology import skeletonize
# im = img_as_float(imread('Chapter06\CH06images\dynasaur.png')[...,3])
# threshold = 0.5
# im[im <= threshold] = 0
# im[im > threshold] = 1
# skeleton = skeletonize(im)
# plot_images_horizontally(im, skeleton, 'skeleton',sz=(18,9))


# # 凸包

# from skimage.morphology import convex_hull_image
# im = rgb2gray(imread('Chapter06\CH06images\horse-dog.jpg'))
# threshold = 0.5
# im[im < threshold] = 0 # convert to binary image
# im[im >= threshold] = 1
# chull = convex_hull_image(im)
# plot_images_horizontally(im, chull, 'convex hull', sz=(18,9))

# # 二值图像和凸包图像的差值图像
# im = im.astype(np.bool)
# chull_diff = img_as_float(chull.copy())
# chull_diff[im] = 2
# # pylab.figure(figsize=(20,10))
# pylab.imshow(chull_diff, cmap=pylab.cm.gray, interpolation='nearest')
# pylab.title('Difference Image', size=8)
# pylab.show()

# # 删除小于指定阈值的对象
# from skimage.morphology import remove_small_objects
# im = rgb2gray(imread('Chapter06\CH06images\circles.jpg'))
# im[im > 0.5] = 1 # create binary image by thresholding with fixed threshold
# 0.5
# im[im <= 0.5] = 0
# im = im.astype(np.bool_)
# # pylab.figure(figsize=(20,20))
# pylab.subplot(2,2,1), plot_image(im, 'original')
# i = 2
# for osz in [50, 200, 500]:
#     im1 = remove_small_objects(im, osz, connectivity=1)
#     pylab.subplot(2,2,i), plot_image(im1, 'removing small objects below size ' + str(osz))
#     i += 1
# pylab.show()

# # 白顶帽与黑顶帽
# from skimage.morphology import white_tophat, black_tophat, square
# im = imread('Chapter06\CH06images\\tagore.png')[...,3]
# im[im <= 0.5] = 0
# im[im > 0.5] = 1
# im1 = white_tophat(im, square(5))
# im2 = black_tophat(im, square(5))
# # pylab.figure(figsize=(20,15))
# pylab.subplot(1,2,1), plot_image(im1, 'white tophat')
# pylab.subplot(1,2,2), plot_image(im2, 'black tophat')
# pylab.show()

# # 提取边界
# from skimage.morphology import binary_erosion
# im = rgb2gray(imread('Chapter06\CH06images\horse-dog.jpg'))
# threshold = 0.5
# im[im < threshold] = 0
# im[im >= threshold] = 1
# boundary = im - binary_erosion(im)
# plot_images_horizontally(im, boundary, 'boundary',sz=(18,9))

# # 利用开闭运算实现指纹图像清洗

# im = rgb2gray(imread('Chapter06\CH06images\\fingerprint.jpg'))
# im[im <= 0.5] = 0 # binarize
# im[im > 0.5] = 1
# im_o = binary_opening(im, square(2))
# im_c = binary_closing(im, square(2))
# im_oc = binary_closing(binary_opening(im, square(2)), square(2))
# # pylab.figure(figsize=(20,20))
# pylab.subplot(221), plot_image(im, 'original')
# pylab.subplot(222), plot_image(im_o, 'opening')
# pylab.subplot(223), plot_image(im_c, 'closing')
# pylab.subplot(224), plot_image(im_oc, 'opening + closing')
# pylab.show()


# # 灰度级操作
# from skimage.morphology import dilation, erosion, closing, opening, square
# im = rgb2gray(imread('Chapter06\CH06images\zebra.jpg'))
# struct_elem = square(5)
# eroded = erosion(im, struct_elem)
# plot_images_horizontally(im, eroded, 'erosion')

# dilated =dilation(im, struct_elem)
# plot_images_horizontally(im, dilated,  'dilation')

# opened = opening(im, struct_elem)
# plot_images_horizontally(im, opened, 'opening')

# closed = closing(im, struct_elem)
# plot_images_horizontally(im, closed, 'closing')

# # 形态学对比度增强

# from skimage.filters.rank import enhance_contrast
# from skimage import exposure
# from skimage.util import img_as_ubyte
# def plot_gray_image(ax, image, title):
#     ax.imshow(image, cmap=pylab.cm.gray),
#     ax.set_title(title), ax.axis('off')
#     ax.set_adjustable('box')
    
# image = rgb2gray(imread('Chapter06\CH06images\squirrel.jpg'))
# sigma = 0.05
# noisy_image = np.clip(image + sigma * np.random.standard_normal(image.shape), 0, 1)
# enhanced_image = enhance_contrast(img_as_ubyte(noisy_image), disk(5))
# equalized_image = exposure.equalize_adapthist(noisy_image)

# fig, axes = pylab.subplots(1, 3, figsize=[18, 7], sharex='row',sharey='row')
# axes1, axes2, axes3 = axes.ravel()
# plot_gray_image(axes1, noisy_image, 'Original')
# plot_gray_image(axes2, enhanced_image, 'Local morphological contrast enhancement')
# plot_gray_image(axes3, equalized_image, 'Adaptive Histogram equalization')

# # 计算局部熵
# from skimage.morphology import disk
# from skimage.filters.rank import entropy
# image = rgb2gray(imread('Chapter06\CH06images\zebra.jpg'))
# fig, (axes1, axes2) = pylab.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)
# fig.colorbar(axes1.imshow(image, cmap=pylab.cm.gray), ax=axes1)
# axes1.axis('off'), axes1.set_title('Image', size=8), axes1.set_adjustable('box')
# fig.colorbar(axes2.imshow(entropy(image, disk(5)), cmap=pylab.cm.inferno), ax=axes2)
# axes2.axis('off'), axes2.set_title('Entropy', size=8), axes2.set_adjustable('box')
# pylab.show()

# # 计算形态学Beucher梯度
# from scipy import ndimage
# im = imread('Chapter06\CH06images\einstein.jpg')
# im_d = ndimage.grey_dilation(im, size=(3,3))
# im_e = ndimage.grey_erosion(im, size=(3,3))
# im_bg = im_d - im_e
# im_g = ndimage.morphological_gradient(im, size=(3,3))
# pylab.gray()
# # pylab.figure(figsize=(20,18))
# pylab.subplot(231), pylab.imshow(im), pylab.title('original', size=8),
# pylab.axis('off')
# pylab.subplot(232), pylab.imshow(im_d), pylab.title('dilation', size=8),
# pylab.axis('off')
# pylab.subplot(233), pylab.imshow(im_e), pylab.title('erosion', size=8),
# pylab.axis('off')
# pylab.subplot(234), pylab.imshow(im_bg), pylab.title('Beucher gradient (bg)', size=8), pylab.axis('off')
# pylab.subplot(235), pylab.imshow(im_g), pylab.title('ndimage gradient (g)', size=8), pylab.axis('off')
# pylab.subplot(236), pylab.title('diff gradients (bg - g)', size=8), pylab.imshow(im_bg - im_g) 
# pylab.axis('off')
# pylab.show()

# 形态学laplace计算

from scipy import ndimage
im = imread('Chapter06\CH06images\\tagore.png')[...,3]
im_g = ndimage.morphological_gradient(im, size=(3,3))
im_l = ndimage.morphological_laplace(im, size=(5,5))
# pylab.figure(figsize=(15,10))
pylab.subplot(121), pylab.title('ndimage morphological laplace', size=8)
pylab.imshow(im_l)
pylab.axis('off')
pylab.subplot(122), pylab.title('ndimage morphological gradient', size=8),
pylab.imshow(im_g)
pylab.axis('off')
pylab.show()
