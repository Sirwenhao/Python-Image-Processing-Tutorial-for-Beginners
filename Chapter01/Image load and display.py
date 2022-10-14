# 四种读取图像的方式

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PIL.ImageChops import add, subtract, multiply, difference,screen
import PIL.ImageStat as stat
from skimage.io import imread, imsave, imshow, show, imread_collection,imshow_collection
from skimage import color, viewer, exposure, img_as_float, data
from skimage.transform import SimilarityTransform, warp, swirl
from skimage.util import invert, random_noise, montage
import matplotlib.pylab as plt
from scipy.ndimage import affine_transform, zoom
from scipy import misc

# 使用PIL读取、保存和显示图像
im = Image.open('Chapter01\images\parrot.jpeg')
print(im.width, im.height, im.mode, im.format, type(im))
# 5464 8192 RGB JPEG <class 'PIL.JpegImagePlugin.JpegImageFile'>
im.show()
im_g = im.convert('L')
im_g.save('Chapter01\images\parrot_gray.png')
Image.open('Chapter01\images\parrot_gray.png').show()