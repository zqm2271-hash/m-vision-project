# Proj 5: Multi-view Geometry

## 基于 Dinosaur Dataset 的多视角三维重建

### 1. 项目简介

本实验使用 Oxford VGG 经典 Dinosaur Dataset，实现基于多视角二维图像的目标三维结构恢复。实验不直接使用数据集提供的相机矩阵作为结果，而是利用多幅图像中的二维特征点轨迹，完成基础矩阵估计、本质矩阵计算、相机姿态恢复和三维点重建。

实验流程属于典型的增量式 Structure from Motion。首先选择一对具有足够共同观测和视差的图像作为初始视角，恢复两相机之间的相对位姿并三角化初始三维点。随后对其他图像，利用已经重建出的三维点和该图像中的二维观测，通过 PnP 估计新相机位姿，并继续三角化更多点，逐步得到 Dinosaur 的稀疏三维点云。

### 2. 实验环境

```text
Python 3
OpenCV
NumPy
Matplotlib
PyYAML
```

虚拟环境位于：

```powershell
D:\lyxxx\5\.venv
```

运行命令：

```powershell
python main.py --download-dinosaur --max-views 10
```

数据已经下载后可以使用：

```powershell
python main.py --dinosaur --max-views 10
```

### 3. 数据集说明

实验采用 Oxford VGG Multi-view Data 中的 Dinosaur Dataset。该数据集包含 36 帧 Dinosaur 图像，以及跨多视角的二维 tracked points 文件 `viff.xy`。

`viff.xy` 的每一行对应一个被跟踪的物体点，每一列对表示该点在某一帧中的二维坐标：

```text
x1 y1 x2 y2 ... x36 y36
```

如果某个点在某一视角中不可见，则该视角的坐标记录为：

```text
-1 -1
```

因此，该文件可以看作已经完成跨视角匹配的特征点集合。

### 4. 算法原理

#### 4.1 特征点匹配关系

传统 SfM 流程需要先对每张图像提取 SIFT、ORB 等局部特征，再进行两两匹配。Dinosaur Dataset 已经提供 tracked points，因此本实验直接使用这些跨视角二维轨迹作为匹配点。这样可以减少特征误匹配对实验的干扰，更集中地验证多视图几何流程。

#### 4.2 基础矩阵 F

对于两幅图像中的匹配点 `x1` 和 `x2`，基础矩阵满足极线约束：

```text
x2^T F x1 = 0
```

程序使用 RANSAC 估计基础矩阵，剔除错误匹配和不稳定观测。代码中对应函数为：

```text
cv2.findFundamentalMat
```

#### 4.3 本质矩阵 E

基础矩阵描述的是像素坐标下的极线几何。本质矩阵描述归一化相机坐标下的极线几何。若相机内参矩阵为 `K`，则：

```text
E = K^T F K
```

由于 Dinosaur 数据集没有在 tracked points 文件中直接提供内参，本实验采用课程实验中常见的近似内参模型：

```text
K = [ f  0  cx
      0  f  cy
      0  0   1 ]
```

其中主点设为图像中心，焦距设为 `1.2 * max(width, height)`。该假设足以完成稀疏结构恢复和姿态链路验证。

#### 4.4 相机姿态恢复

本质矩阵可以分解为旋转矩阵和平移方向：

```text
E = [t]x R
```

程序调用 OpenCV 的：

```text
cv2.recoverPose
```

从 `E` 中恢复初始两帧之间的相对姿态 `R, t`。其中 `t` 只能恢复方向，尺度在单目 SfM 中不可确定。

#### 4.5 三角化三维点

已知两个相机投影矩阵：

```text
P1 = K [I | 0]
P2 = K [R | t]
```

对两视角匹配点进行三角化，可以恢复三维点的齐次坐标。代码中使用：

```text
cv2.triangulatePoints
```

得到初始稀疏三维点云。

#### 4.6 增量式多视角重建

完成初始两视角重建后，程序继续注册新的图像：

1. 查找新图像中已经存在三维点的二维观测；
2. 使用 `cv2.solvePnPRansac` 估计该图像相机位姿；
3. 将新图像与已注册图像中共同可见但尚未重建的点进行三角化；
4. 将新三维点加入点云；
5. 重复上述过程，逐步扩展三维结构。

该方法就是简化版的增量式 SfM。

### 5. 代码结构

```text
main.py
src/dinosaur.py
src/geometry.py
src/sfm.py
src/reconstruction.py
src/visualization.py
```

各文件作用：

```text
main.py              程序入口，负责参数解析和流程调度
dinosaur.py          Dinosaur 数据下载、图片读取、tracked points 解析
geometry.py          基础矩阵、本质矩阵、位姿恢复、PnP、三角化
sfm.py               增量式 SfM 主流程
reconstruction.py    点云上色和 PLY 文件保存
visualization.py     极线图、点云图和相机轨迹图保存
```

### 6. 实验结果

运行：

```powershell
python main.py --dinosaur --max-views 10
```

终端输出：

```text
Dataset images: 36
Tracked feature points: 4983
Initial pair: frame 25 and frame 27
Initial pose inliers: 206
Registered views: [25, 27, 26, 28, 29, 24, 30, 23, 31, 22]
Sparse 3D points: 1313
```

初始两视角估计得到的基础矩阵：

```text
F =
[[-0.     -0.      0.001 ]
 [ 0.     -0.      0.0197]
 [-0.0036 -0.019   0.9996]]
```

根据 `E = K^T F K` 得到本质矩阵：

```text
E =
[[ -0.4718  -2.9348  -0.3544]
 [  2.1608  -0.3812  17.9368]
 [ -2.5733 -17.6411  -0.0067]]
```

恢复得到的相机相对旋转和平移方向：

```text
R =
[[ 0.9856 -0.1637  0.0435]
 [ 0.1638  0.9865  0.0012]
 [-0.0432  0.0059  0.9991]]

t = [-0.9862 -0.0194  0.1645]
```

输出文件：

```text
outputs/dinosaur_sparse.ply
outputs/dinosaur_reconstruction.png
outputs/dinosaur_epipolar_lines.png
outputs/dinosaur_official_dense.ply
outputs/dinosaur_official_reconstruction.png
outputs/dinosaur_three_views.png
```

其中 `dinosaur_sparse.ply` 是自己估计 F/E/R/t 后生成的稀疏点云；`dinosaur_official_dense.ply` 是使用官方 camera matrices 对全部 tracked points 做多视角三角化后的优化点云，可用 MeshLab 或 CloudCompare 打开查看。`dinosaur_three_views.png` 从三个方向显示点云投影，能更清楚观察 Dinosaur 的身体、头部、腿和尾巴。

### 7. 代码层面的优化说明

上一版点云不容易看出 Dinosaur，主要原因是只使用近似内参和少量注册视角，且没有 bundle adjustment，三维点会有较多尺度和姿态误差。优化时，代码中新增了 `src/multiview.py`，使用官方提供的 `dino_Ps.mat` 作为高质量投影矩阵，对所有 tracked points 做多视角三角化。

具体实现如下：

1. 在 `src/dinosaur.py` 中，`download_dinosaur_dataset()` 额外下载：

```text
https://www.robots.ox.ac.uk/~vgg/data/dino/dino_Ps.mat
```

并由 `load_camera_matrices()` 使用 `scipy.io.loadmat()` 读取其中的 3x4 投影矩阵。

2. 在 `src/multiview.py` 中，`reconstruct_from_official_cameras()` 遍历 `viff.xy` 每一行。每一行是同一个三维点在 36 帧中的二维观测。代码会跳过 `-1 -1` 缺失点，只保留至少出现在 4 个视角中的 track。

3. 对每个 track，`triangulate_n_view()` 构造多视角 DLT 线性方程。对于第 `i` 个视角中的观测点 `(x_i, y_i)` 和投影矩阵 `P_i`，加入两行约束：

```text
x_i * P_i[2] - P_i[0]
y_i * P_i[2] - P_i[1]
```

所有视角的约束组成矩阵 `A`，然后执行：

```text
_, _, vt = np.linalg.svd(A)
X = vt[-1]
X = X[:3] / X[3]
```

得到三维点坐标。

4. `mean_reprojection_error()` 会把得到的三维点重新投影到所有可见图像中：

```text
x_proj = P_i X
x_proj = x_proj[:2] / x_proj[2]
```

然后计算投影点和原始 tracked point 之间的平均像素误差。代码只保留平均重投影误差小于 `2.5 px` 的点，因此可以去掉大量漂浮点。

5. `src/reconstruction.py` 中的 `colorize_tracks()` 根据每个 track 第一次可见图像上的像素颜色给三维点上色，并由 `save_ply()` 输出彩色 PLY 点云。

6. `src/visualization.py` 中的 `save_dinosaur_views()` 把三维点云分别投影到 `X-Z`、`X-Y`、`Y-Z` 平面，生成 `dinosaur_three_views.png`。这个图比单个 3D 视角更容易看出 Dinosaur 轮廓。

### 8. 结果分析

实验成功从 Dinosaur 的多视角二维 tracked points 中恢复出了相机相对姿态和稀疏三维结构。初始两帧通过 RANSAC 估计基础矩阵，能够剔除不稳定匹配点；由本质矩阵恢复的 `R, t` 提供了三角化所需的相机投影矩阵；后续视角通过 PnP 加入重建，使点云规模逐步增加。

自己估计位姿的点云用于展示完整几何流程；官方相机矩阵辅助的点云用于优化最终可视化。由于官方投影矩阵来自更完整的三维重建结果，多视角三角化后的平均重投影误差约为 `0.348 px`，因此恐龙轮廓比上一版更清楚。

### 9. 总结

本项目完成了多视图几何作业要求的完整流程：

```text
多视角二维点匹配
-> 基础矩阵 F
-> 本质矩阵 E
-> 相机姿态 R, t
-> 三角化三维点
-> 增量式多视角稀疏三维重建
```

实验基于 Oxford VGG Dinosaur Dataset，最终输出 Dinosaur 目标物体的稀疏三维点云，实现了从多幅二维图像恢复三维结构的目标。
