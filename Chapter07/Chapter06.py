# 2022/12/01  author:WH

# scikit-image模块中的corner_harris()函数
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from matplotlib import pylab
from skimage.feature import corner_harris, corner_peaks, corner_subpix

# image = imread('Chapter07\CH07images\chess_football.png')
# image1 = cv2.imread('Chapter07\CH07images\chess_football.png')
# image_gray = cv2.imread('Chapter07\CH07images\chess_football.png', cv2.IMREAD_GRAYSCALE)
# coordinates = corner_harris(image_gray, k=0.001)
# image[coordinates > 0.01*coordinates.max()] = [255, 0, 0, 255]
# pylab.figure(figsize=(20, 10))
# pylab.imshow(image), pylab.axis('off'),pylab.show()

# # 对角点的精细化操作

# image = imread('Chapter07\CH07images\pyramids2.jpg')
# image_gray = rgb2gray(image)
# coordinates = corner_harris(image_gray, k=0.001)
# coordinates[coordinates > 0.03*coordinates.max()] = 255
# corner_coordinates = corner_peaks(coordinates)
# coordinates_subpix = corner_subpix(image_gray, corner_coordinates, window_size=11)
# # pylab.figure(figsize=(20, 20))
# pylab.subplot(211), pylab.imshow(coordinates, cmap='inferno')
# pylab.plot(coordinates_subpix[:,1], coordinates_subpix[:, 0], 'r.', markersize=5, label='subpixel')
# pylab.legend(prop={'size':8}), pylab.axis('off')
# pylab.subplot(212), pylab.imshow(image, interpolation='nearest')
# pylab.plot(corner_coordinates[:, 1], corner_coordinates[:, 0], 'bo', markersize=5)
# pylab.plot(coordinates_subpix[:, 1], coordinates_subpix[:, 0], 'r+', markersize=5), pylab.axis('off')
# pylab.tight_layout(), pylab.show()


# 使用RANSAC(随机抽样一致性算法)和哈里斯角点特征实现鲁棒图像匹配
from skimage.util import img_as_float
import numpy as np
from skimage.measure import ransac
from skimage.exposure import rescale_intensity
from skimage.transform import warp,SimilarityTransform, AffineTransform, resize

# temple = rgb2gray(img_as_float(imread('Chapter07\CH07images\\temple.jpg')))
# image_original = np.zeros(list(temple.shape) + [3])
# image_original[..., 0] = temple
# gradient_row, gradient_col = (np.mgrid[0:image_original.shape[0], 0:image_original.shape[1]] / float(image_original.shape[0]))
# image_original[..., 1] = gradient_row
# image_original[..., 2] = gradient_col
# image_original = rescale_intensity(image_original)
# image_original_gray = rgb2gray(image_original)
# affine_trans = AffineTransform(scale=(0.8, 0.9), rotation=0.1, translation=(120, -20))
# image_warped = warp(image_original, affine_trans.inverse, output_shape=image_original.shape)
# image_warped_gray = rgb2gray(image_warped)
# coordinates = corner_harris(image_original_gray)
# coordinates[coordinates > 0.01*coordinates.max()] = 1
# coordinates_original = corner_peaks(coordinates, threshold_rel=0.0001, min_distance=5)
# coordinates = corner_harris(image_warped_gray)
# coordinates[coordinates > 0.01*coordinates.max()] = 1
# coordinates_warped = corner_peaks(coordinates, threshold_rel=0.0001, min_distance=5)
# coordinates_original_subpix = corner_subpix(image_original_gray, coordinates_original, window_size=9)
# coordinates_warped_subpix = corner_subpix(image_warped_gray, coordinates_warped, window_size=9)

# def gaussian_weights(window_ext, sigma=1):
#     y, x = np.mgrid[-window_ext:window_ext+1, -window_ext:window_ext+1]
#     g_w = np.zeros(y.shape, dtype = np.double)
#     g_w[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
#     g_w /= 2 * np.pi * sigma * sigma
#     return g_w

# def match_corner(coordinates, window_ext=3):
#     row, col = np.round(coordinates).astype(np.intp)
#     window_original = image_original[row-window_ext:row+window_ext+1, col-window_ext:col+window_ext+1, :]
#     weights = gaussian_weights(window_ext, 3)
#     weights = np.dstack((weights, weights, weights))
#     SSDs = []
#     for coord_row, coord_col in coordinates_warped:
#         window_warped = image_warped[coord_row-window_ext:coord_row+window_ext+1,
#         coord_col-window_ext:coord_col+window_ext+1, :]
#         if window_original.shape == window_warped.shape:
#             SSD = np.sum(weights * (window_original - window_warped)**2)
#             SSDs.append(SSD)
#     min_idx = np.argmin(SSDs) if len(SSDs) > 0 else -1
#     return coordinates_warped_subpix[min_idx] if min_idx >= 0 else [None]

# from skimage.feature import (match_descriptors, corner_peaks, corner_harris, plot_matches, BRIEF)
# source, destination = [], []
# for coordinates in coordinates_original_subpix:
#     coordinates1 = match_corner(coordinates)
#     if any(coordinates1) and len(coordinates1) > 0 and not all(np.isnan(coordinates1)):
#         source.append(coordinates)
#         destination.append(coordinates1)
# source = np.array(source)
# destination = np.array(destination)
# model = AffineTransform()
# model.estimate(source, destination)
# model_robust, inliers = ransac((source, destination), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100)
# outliers = inliers == False
# print(affine_trans.scale, affine_trans.translation, affine_trans.rotation)
# print(model.scale, model.translation, model.rotation)
# print(model_robust.scale, model_robust.translation, model_robust.rotation)

# fig, axes = pylab.subplots(nrows=2, ncols=1, figsize=(20,15))
# pylab.gray()
# inlier_idxs = np.nonzero(inliers)[0]
# plot_matches(axes[0], image_original_gray, image_warped_gray, source, destination, np.column_stack((inlier_idxs, inlier_idxs)),matches_color='b')
# axes[0].axis('off'), axes[0].set_title('Correct correspondences', size=8)
# outlier_idxs = np.nonzero(outliers)[0]
# plot_matches(axes[1], image_original_gray, image_warped_gray, source, destination, np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
# axes[1].axis('off'), axes[1].set_title('Faulty correspondences', size=8)
# fig.tight_layout(), pylab.show()

# # 基于LoG、DoG和DoH的斑点检测
# from numpy import sqrt
# from skimage.feature import blob_dog, blob_log, blob_doh
# im = cv2.imread('Chapter07\CH07images\\butterfly.png')
# im_gray = rgb2gray(im)
# log_blobs = blob_log(im_gray, max_sigma=30, num_sigma=10, threshold=0.1)
# log_blobs[:, 2] = sqrt(2) * log_blobs[:, 2] # Compute radius in the 3rd column
# dog_blobs = blob_dog(im_gray, max_sigma=30, threshold=0.1)
# dog_blobs[:, 2] = sqrt(2) * dog_blobs[:, 2]
# doh_blobs = blob_doh(im_gray, max_sigma=30, threshold=0.005)
# list_blobs = [log_blobs, dog_blobs, doh_blobs]
# colors, titles = ['yellow', 'lime', 'red'], ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
# sequence = zip(list_blobs, colors, titles)
# fig, axes = pylab.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
# axes = axes.ravel()
# axes[0].imshow(im, interpolation='nearest')
# axes[0].set_title('original image', size=10), axes[0].set_axis_off()
# for idx, (blobs, color, title) in enumerate(sequence):
#     axes[idx+1].imshow(im, interpolation='nearest')
#     axes[idx+1].set_title('Blobs with ' + title, size=10)
#     for blob in blobs:
#         y, x, row = blob
#         col = pylab.Circle((x, y), row, color=color, linewidth=2, fill=False)
#         axes[idx+1].add_patch(col), axes[idx+1].set_axis_off()
# pylab.show()

# 基于scikit-image特征模块的hog()函数计算HOG描述符并可视化
from skimage.feature import hog
from skimage import exposure
image = rgb2gray(imread('Chapter07\CH07images\cameraman.jpg'))
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1),visualize=True)
print(image.shape, len(fd))
fig, (axes1, axes2) = pylab.subplots(1,2,figsize=(15,10),sharex=True,sharey=True)
axes1.axis('off'), axes1.imshow(image, cmap=pylab.cm.gray),axes1.set_title('Input image')
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
axes2.axis('off'),axes2.imshow(hog_image_rescaled, cmap=pylab.cm.gray),axes2.set_title('Histogram of Oriented Gradients')
pylab.show()
