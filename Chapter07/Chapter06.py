# 2022/12/01  author:WH

# scikit-image模块中的corner_harris()函数
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from matplotlib import pylab
from skimage.feature import corner_harris

image = imread('Chapter07\CH07images\chess_football.png')
image1 = cv2.imread('Chapter07\CH07images\chess_football.png')
image_gray = cv2.imread('Chapter07\CH07images\chess_football.png', cv2.IMREAD_GRAYSCALE)
coordinates = corner_harris(image_gray, k=0.001)
image[coordinates > 0.01*coordinates.max()] = [255, 0, 0, 255]
pylab.figure(figsize=(20, 10))
pylab.imshow(image), pylab.axis('off'),pylab.show()



