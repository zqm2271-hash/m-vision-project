# Proj 5: Multi-view Geometry - Dinosaur SfM

本项目完成基于多视角二维图像的目标物体三维结构恢复。算法采用增量式 Structure from Motion 思路，数据集使用 Oxford VGG 的经典 Dinosaur Dataset。

## 1. 实现目标

输入 Dinosaur 数据集中的多幅二维图像和跨视角特征点轨迹，完成：

- 根据两视角匹配点估计基础矩阵 `F`
- 根据相机内参 `K` 计算本质矩阵 `E = K^T F K`
- 从 `E` 恢复相机相对姿态 `R, t`
- 对匹配点进行三角化，得到初始三维点
- 使用 PnP 注册后续视角，继续三角化新点
- 输出 Dinosaur 的稀疏三维点云结构

## 2. 数据集

数据集：Oxford VGG Multi-view Data - Dinosaur

官方页面：

```text
https://www.robots.ox.ac.uk/~vgg/data/mview/
```

Dinosaur 数据包含：

- 36 帧 Dinosaur 图像
- 跨 36 帧的二维 tracked points 文件 `viff.xy`
- 每行表示同一个物体点在多幅图像中的二维观测，缺失观测用 `-1 -1` 表示

本项目使用 `viff.xy` 作为已经匹配好的特征点轨迹，不直接使用官方给出的相机矩阵作为结果。

## 3. 环境配置

虚拟环境已创建在：

```powershell
D:\lyxxx\5\.venv
```

在 VS Code 终端中激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果重新创建环境，安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 4. 运行方式

首次下载 Dinosaur 数据集并运行：

```powershell
python main.py --download-dinosaur --max-views 10
```

数据已经下载后再次运行：

```powershell
python main.py --dinosaur --max-views 10
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

`dinosaur_sparse.ply` 是自己估计 F/E/R/t 后得到的 SfM 链路点云。`dinosaur_official_dense.ply` 使用官方 camera matrices 对全部可见视角做多视角三角化，并通过重投影误差过滤，轮廓更清楚，适合作为最终展示图。PLY 可以用 MeshLab、CloudCompare、Open3D 等工具查看。

## 5. 代码结构

```text
main.py
src/
  dinosaur.py         # Dinosaur 数据下载、读取、tracked points 解析
  geometry.py         # F、E、R/t、PnP、三角化
  sfm.py              # 增量式 SfM 重建流程
  multiview.py        # 官方相机矩阵辅助的多视角 DLT 三角化
  reconstruction.py   # 点云颜色绑定和 PLY 保存
  visualization.py    # 极线图、点云图、相机轨迹图
```

## 6. 代码层面的优化实现

为了让点云更像 Dinosaur，本项目新增了 `src/multiview.py`：

- `load_camera_matrices()` 从 `data/dinosaur/dino_Ps.mat` 读取官方 3x4 投影矩阵 `P`
- `reconstruct_from_official_cameras()` 遍历 `viff.xy` 中每条 track，收集该点在所有可见视角中的二维观测
- `triangulate_n_view()` 对每个点构造多视角 DLT 方程：

```text
x_i * P_i[2] - P_i[0]
y_i * P_i[2] - P_i[1]
```

然后用 SVD 求三维齐次点。

- `mean_reprojection_error()` 将三维点重新投影回所有可见图像，计算平均像素误差
- 只保留重投影误差小于阈值的点，减少漂浮点
- `save_dinosaur_views()` 输出正视、俯视、侧视三张二维投影图，让恐龙轮廓更容易看出来

## 7. 当前验证结果

在本机运行：

```powershell
python main.py --dinosaur --max-views 10
```

得到：

```text
Dataset images: 36
Tracked feature points: 4983
Initial pair: frame 25 and frame 27
Initial pose inliers: 206
Registered views: [25, 27, 26, 28, 29, 24, 30, 23, 31, 22]
Sparse 3D points: 1313
Official-camera filtered 3D points: 1400
Mean reprojection error: 0.348 px
```

说明程序已经完成从多视角二维点轨迹到 Dinosaur 稀疏三维结构的重建。
