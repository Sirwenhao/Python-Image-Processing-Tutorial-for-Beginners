### 《Python图像处理实战》Chapter03

- 卷积定理和频域高斯模糊
- 频域滤波

核心操作是利用傅里叶变换时域与频域之间的变化特性，空域卷积对应频域乘积。频域滤波的基本步骤包含一下几个部分：

- 原始图像与卷积核进行DFT变换
- 频域乘积
- IDFT变换得到重建图像

#### 图像信噪比的定义

信号均值与背景标准差的比值：
$$
SNR=(\frac{\mu_{sig}}{\delta_{bg}})
$$
 对于图像，**这里的”信号值“往往是灰度值**。分母有时采用背景信号值的方差，代表的物理意义是噪声功率。对于高对比度黑背景图，上式直接计算的结果通常是无穷。所以我们改用信号均值与信号标准偏差来衡量

图像信噪比函数实现：

```python
import numpy as np
def signaltonoise(im, axis=0, ddof=0):
    im = np.asanyarray(im)
    avg = im.mean(axis)
    std = im.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, avg/std)
```

Reference:

- `np.asanyarray(arr, dtype=None, order=None)`函数将输入转换成数组，具体用法：
  - `arr`：输入数据，可以转换为数组的任何形式。这包括标量、列表、元组列表、元组、元组的元组、列表的元组和ndarray
  - 是使用行为主(c风格)还是列为主(fortran风格)的内存表示。默认为“C”
  - 返回:[ndarray或一个ndarray子类]arr的数组解释。如果arr是ndarray或ndarray的子类，它将按原样返回，并且不执行复制操作
- `np.mean(a, axis, dtype, out, keepdims)`求平均值，参数axis的解释：
  - axis不设置值，对a中的所有数求均值，返回一个数值
  - axis=0，压缩行，对各列求均值，返回$1*n$的矩阵
  - axis=1，压缩列，对各行求均值，返回$m*1$的矩阵

代码示例：

```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
print(np.mean(a))
2.5
print(np.mean(a, axis=0))
[2. 3.]
print(np.mean(a, axis=1))
[1.5 3.5]
```

- `np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=)`函数用于求标准差，具体用法：
  - axis=0时，表示求每一列标准差
  - axis=1时，表示求每一行标准差
  - 当axis=None时，表示求全局标准差
  - 当ddof=0时，计算有偏样本标准差；一般在拥有所有数据的情况下，计算所有数据的标准差时使用，即最终除以n
  - 当ddof = 1时，表示计算无偏样本标准差，最终除以n-1

代码示例：

```python
import numpy as np
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
pian = np.std(a, ddof = 0) # 有偏
print("std有偏计算结果：",pian)
std有偏计算结果： 2.8722813232690143

orig = np.sqrt(((a - np.mean(a)) ** 2).sum() / a.size)
print("有偏公式计算结果：",orig)
有偏公式计算结果： 2.8722813232690143

no_pian = np.std(a, ddof = 1) # 无偏
print("std无偏计算结果：",no_pian)
std无偏计算结果： 3.0276503540974917

orig1 = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
print("无偏公式计算结果：",orig1)
无偏公式计算结果： 3.0276503540974917
```



#### PIL库中的convert()函数

`img.convert(mode=None, matrix=None, dither=None, palette=0, colors=256)`，PIL中有9种不同的模式，具体如下：

| modes |                 描述                  |
| :---: | :-----------------------------------: |
|   1   |    1位像素，黑和白，存成8位的像素     |
|   L   |             8位像素，黑白             |
|   P   | 8位像素，使用调色板映射到任何其他模式 |
|  RGB  |           3× 8位像素，真彩            |
| RGBA  |       4×8位像素，真彩+透明通道        |
| CMYK  |          4×8位像素，颜色隔离          |
| YCbCr |        3×8位像素，彩色视频格式        |
|   I   |             32位整型像素              |
|   F   |            32位浮点型像素             |

