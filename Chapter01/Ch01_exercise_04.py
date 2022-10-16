# 2022/10/16 author:WH 
# 使用scikit-image的warp()函数实现旋流变换
import numpy as np
from skimage.io import imread
from skimage.transform import warp
import matplotlib.pylab as plt

def swirl(xy, x0, y0, R):
    r = np.sqrt((xy[:, 1]-x0)**2 + (xy[:, 0]-y0)**2)
    a = np.pi*r/R
    xy[:, 1] = (xy[:, 1]-x0)*np.cos(a) + (xy[:, 0]-y0)*np.sin(a) + x0
    xy[:, 0] = -(xy[:, 1]-x0)*np.sin(a) + (xy[:, 0]-y0)*np.cos(a) + y0
    return xy

im = imread('Chapter01\images\mandrill.jpg')
print(im.shape)
im1 = warp(im, swirl, map_args={'x0':1000, 'y0':1000, 'R':100})
plt.figure(figsize=(2000, 2000))
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('Input image', size=20)
plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('Output image', size=20)
plt.show()
