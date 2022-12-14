### 《Python图像处理实战》Chapter05学习笔记

主要内容：

- 图像导数——图像梯度和拉普拉斯算子
- 锐化和反锐化掩模
- 使用导数和滤波器进行边缘检测
- 图像金字塔——融合图像

图像$I$即函数$(f(x,y))$的梯度的大小对应于图像中边缘部分的强度，图像梯度的方向垂直于边缘。在输入图像中，强度（灰度值）剧烈变化的位置对应于图像一阶导数强度中有尖峰或谷的位置。简单来说，梯度幅值的峰值表示边缘的位置，对梯度幅值设定阈值可以找到图像中的边缘信息。

#### 一阶偏导对应与卷积核的对应关系：

$\frac{\partial f}{\partial x}=f(x+1)-f(x)$对应于卷积核$\begin{bmatrix} -1 & 1 \end{bmatrix}$，$\frac{\partial f}{\partial y}=f(y+1)-f(y)$对应于卷积核$\begin{bmatrix} -1 \\ 1 \end{bmatrix}$，中心差分对应卷积核$\frac{\partial f}{\partial x}=\frac{f(x+1)-f(x-1)}{2}$对应于卷积核$\begin{bmatrix}-1 & 0 & 1 \end{bmatrix}$

#### 拉普拉斯算子

拉普拉斯算子近似于图像的二阶导数，用于检测边缘，零交叉点用于标记边缘位置。

拉普拉斯算子与其对应的卷积核:

$\bigtriangledown^2f=\frac{\partial^{2}f}{\partial^{2}x^2}+\frac{\partial^{2}f}{\partial^{2}y^2}=f(x+1,y)+f(x-1,y)+f(x,y+1),f(x,y-1)-4f(x,y)$对应的卷积核为：$\begin{bmatrix}0 & 1 & 0 \\1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$

对拉普拉斯算子的几个关键点说明：

- $\bigtriangledown^2f$是一个标量，也就意味之拉普拉斯算子没有任何的方向信息
- 拉普拉斯算子对于噪声非常敏感，因此在使用拉普拉斯算子之前需要对其进行平滑处理（例如使用高斯滤波器滤波）

#### 锐化和反锐化掩模

锐化的目的是图像的细节信息或增强模糊的细节

##### 使用拉普来滤波器来锐化图像的一般步骤

1. 对原始输入图像应用拉普拉斯滤波器
2. 将步骤一得到的图像与原始图像进行叠加(所得即锐化后的图像)

##### 反锐化掩模

反锐化掩模是一种用于锐化图像的技术，即从图像本身减去图像的模糊版本，用于反锐化掩模的典型混合公式如下：
$$
锐化图像=原始图像+(原始图像-模糊图像)\times 总数
$$
锐化后的图像与原始图像机器细节图像的关系：
$$
原始图像-模糊图像=细节图像\\
(用高斯滤波器)\\
原始图像+\alpha*(细节图像)=锐化图像
$$

#### 使用导数和滤波器的边缘检测

图像边缘：图像强度函数中突然发生急剧变化（不连续）的像素一般对应于图像的边缘，而边缘检测所要实现的任务，就是找出这些变化剧烈的区域。边缘检测一般是通过设定阈值，对梯度图像进行阈值化处理所得。

##### 常用的一阶二阶图像边缘检测滤波器滤波核：

- 逼近一阶图像导数的滤波器：
  - Sobel滤波器：$$Sobel_x \begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix}$$，$$Sobel_y \begin{bmatrix}1&2&1\\0&0&0\\-1&-2&-1\end{bmatrix}$$
  - Scharr滤波器：$$Scharr_x \begin{bmatrix}3&0&-3\\10&0&-10\\3&0&-3\end{bmatrix}$$，$$Scharr_y \begin{bmatrix}3&10&3\\0&0&0\\-3&-10&-3\end{bmatrix}$$
  - Prewitt滤波器：$$Prewitt_x \begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix}$$，$$Scharr_y \begin{bmatrix}1&1&1\\0&0&0\\1&1&1\end{bmatrix}$$
  - Roberts滤波器：$$Roberts_x \begin{bmatrix}0&1\\-1&\end{bmatrix}$$，$$Roberts_y \begin{bmatrix}1&0\\0&1\end{bmatrix}$$

- 逼近二阶图像导数的滤波器：
  - Laplace滤波器：$$Laplace \begin{bmatrix}0&-1&0\\-1&4&-1\\0&-1&0\end{bmatrix}$$

边缘检测的后处理步骤是非最大抑制，会使得一阶倒数的得到的边缘变薄，上述滤波器还不具备此种功能。Canny算子是一种功能先进、水平顶尖的边缘检测滤波器，可以实现对于滤波后的边缘非最大抑制处理。

##### Canny边缘检测器的一般步骤：

1. 平滑/去噪，边缘检测一般对于噪声较为敏感，因此需要先使用高斯滤波器进行平滑滤波处理
2. 计算梯度的大小和方向，对图像应用Sobel水平滤波器和垂直滤波器，计算每个图像的边缘梯度大小和方向，所计算出的梯度角被四舍五入为四个角中的一个，用以表示每个像素的水平、垂直和两个对角线方向
3. 非最大值抑制，任何未使用的可能不能构成边缘的像素被删除，这一步需要检查它是否是梯度方向上邻域内的局部最大值
4. 链接和滞后阈值，这一步确定所检测到的边缘是否为强边缘。使用两个(滞后)阈值$max\_val$和$min\_val$，边缘强度的梯度值高于$max\_val$被保存，低于$min\_val$被删除。根据阈值之间的连通性，划分边缘和非边缘，如果和"确定边"元素相连，则被认为是边缘的一部分，否则丢弃

此处，所使用的高斯平滑滤波器的强度参数的大小会影响边缘检测的效果。

##### LoG滤波器和DoG滤波器

高斯拉普拉斯滤波器（Laplacian of Gaussian，LoG）是一种线性滤波器，本质是对图像进行高斯滤波后紧接着使用拉普拉斯滤波器的组合。LoG滤波核中每一个位置上的数值大小的计算为：
$$
LoG(x,y) =  \nabla^2 G_{\sigma}(x,y)= \frac{\partial{G_{\sigma}(x,y)}}{\partial{x}} + \frac{\partial{G_{\sigma}(x,y)}}{\partial{y}}=\frac{1}{-\pi*{\sigma}^4}e^{-\frac{x^2+y^2}{2\sigma^2}}(1-\frac{x^2+y^2}{2\sigma^2})
$$
DoG滤波器的近似计算为：
$$
\nabla^2{G_\sigma}\approx G_{\sigma1} - G_{\sigma_2}\\\sigma_1=\sqrt{2}\sigma,\sigma_2=\frac{\sigma}{\sqrt2}
$$
两种滤波器实现代码为：

```python
# 2022/11/5 LoG滤波器实现
def LoG(k=12, s=3):
    n = 2*k+1 # size of kernel
    kernel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kernel[i, j] = -(1-((i-k)**2+(j-k)**2)/(2.*s**2))*np.exp(-((i-k)**2+(j-k)**2)/(2.*s**2))/(pi*s**4)
    kernel = np.round(kernel / np.sqrt((kernel**2).sum()),3)
    return kernel

# 2022/11/5 DoG滤波器实现
def DoG(k=12, s=3):
    n = 2*k+1 # size of the kernel
    s1, s2 = s * np.sqrt(2), s / np.sqrt(2)
    kernel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kernel[i, j] = np.exp(-((i-k)**2+(j-k)**2)/(2.*s1**2))/(2*pi*s1**2)
            - np.exp(-((i-k)**2+(j-k)**2)/(2.*s2**2))/(2*pi*s2**2)
    kernel = np.round(kernel/np.sqrt((kernel**2).sum()), 3)
    return kernel
```

#### 图像金字塔

- 高斯金字塔：首先利用平滑滤波器(高斯滤波器)进行图像平滑操作，然后在每一次迭代前一层图像时进行二次抽样，直至最小分辨率
- 拉普拉斯金字塔：拉普拉斯金字塔可以从高斯金字塔的最小尺寸图象开始，通过本层图像的扩展(上采样加平滑)，将其减去下一层的高斯金字塔图像，重复迭代这个过程直至回复原始图像的大小

##### 高斯金字塔的构建步骤

- 从原始图像开始，使用高斯滤波器滤波平滑图像，然后对图像进行下采样

- 迭代上述步骤，直至图像大小变得足够小(如$1\times1$)的层停止

- 高斯金字塔图像原理图：

  <img src="https://gitee.com/sirwenhao/typora-illustration/raw/master/image-20221124214338479.png" alt="image-20221124214338479" style="zoom:33%;" />

- 实现代码：

  ```python
  # Image Gaussian Pyramid
  import numpy as np
  import matplotlib.pyplot as plt
  from skimage.io import imread
  from skimage.color import rgb2gray
  from skimage.transform import pyramid_reduce, pyramid_laplacian, pyramid_expand, resize
  
  image = imread('')
  
  # Gaussian Pyramid
  def get_gaussian_pyramid(image):
      rows, cols, dim = image.shape
      gaussian_pyramid = [image]
      while rows > 1 and cols > 1:
          image = pyramid_reduce(image, downscale=2)
          gaussian_pyramid.append(image)
          rows //= 2
          cols //= 2
      return gaussian_pyramid
  
  # Laplacian Pyramid
  def get_laplacian_pyramid(image):
      # 拉普拉斯金字塔是从高斯金字塔的最小值开始的，所以保留高斯金字塔的最小值
      laplacian_pyramid = [gaussian_pyramid[len(gaussian_pyramid)-1]]
      for i in range(len(gaussian_pyramid-2, -1, -1)):
          # 使用skimage中的pyramid_expand进行上采样，并限定为指定shape
          iamge = gaussian_pyramid[i] - resize(pyramid_expand(gaussian_pyramid[i+1]), gaussian_pyramid[i].shape)
          laplacian_pyramid.append(np.copy(image))
      laplacian_pyramid = laplacian_pyramid[::-1]
      return laplacian_pyramid
  
  # 应用
  gaussian_pyramid = get_gaussian_pyramid(image)
  laplacian_pyramid = get_laplacian_pyramid(gaussian_pyramid)
  
  w, h = 20, 12
  for i in range(3):
      plt.figure(figsize=(w,h))
      p = gaussian_pyramid[i]
      plt.imshow(p)
      plt.title(str(p.shape[0]) + 'x' + str(p.shape[1]), size=20)
      plt.axis('off')
      w, h = w / 2, h / 2
      plt.show()
      
  w, h = 10, 6
  for i in range(1,4):
      plt.figure(figsize=(w,h))
      p = laplacian_pyramid[i]
      plt.imshow(rgb2gray(p), cmap='gray')
      plt.title(str(p.shape[0]) + 'x' + str(p.shape[1]), size=20)
      plt.axis('off')
      w, h = w / 2, h / 2
      plt.show()
  ```

##### 拉普拉斯金字塔构建步骤

- 从高斯金字塔和高斯金字塔的最小图像开始，迭代计算当前层的图像和所获得的图像(先上采样，然后平滑高斯金字塔的上一层图像)之间的差值图像

- 图像大小与原始图像大小相等时终止

- 拉普拉斯金字塔原理图：

  <img src="https://gitee.com/sirwenhao/typora-illustration/raw/master/image-20221124214913188.png" alt="image-20221124214913188" style="zoom:50%;" />

- 具体实现代码见上述代码
