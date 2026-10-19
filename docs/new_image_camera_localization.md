# 新图片在场景中的相机位姿定位

本文说明 StageRecon 当前如何把一张**不在 COLMAP 数据集中的新图片**定位到
Friends Monica apartment 的 3D 场景中。

当前实现对应脚本：

```text
tools/localize_query_from_disk.py
```

## 1. 目标与坐标系

输入是一张新的图片，例如：

```text
outputs/Friends-monica_apartment/new_image_localization/images/query_0001.jpg
```

输出是这张图片在已有 COLMAP 世界坐标中的相机位姿。已有场景来自：

```text
sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/colmap/
```

点云和相机使用同一个坐标系：

```text
raw_colmap_world
```

因此输出的 `camera_center` 可以直接和 `pc.ply` 或语义点云一起在 3D 查看器中显示。

查询图片本身不会被写入 COLMAP，也不会读取查询图片的真值位姿。它的位姿完全由图片特征与已有 3D 点之间的对应关系估计得到。

## 2. 输入数据

定位需要以下数据：

| 数据 | 用途 |
| --- | --- |
| 新图片 | 提供查询图像坐标和查询特征 |
| 查询图的 `keypoints.h5` | 查询图的 DISK 关键点坐标 |
| 查询图的 `descriptors.h5` | 查询图的 DISK 描述子 |
| `retrieval_candidates.tsv` | 为查询图检索出的参考图片列表 |
| 场景 `h5/descriptors.h5` | 已有参考帧的 DISK 描述子 |
| 场景 `colmap/images.bin`、`points3D.bin`、`cameras.bin` | 参考帧的注册信息和 3D 点 |

查询图和参考帧必须使用同一个 DISK 模型和相同的特征格式。当前流程使用 `depth-save.pth` 生成 DISK 特征；如果特征模型、图像缩放方式或描述子格式不一致，匹配数量通常会明显下降。

可选的查询 mask 用于排除人物或其他动态区域。脚本约定 mask 中值为 `0` 的像素保留，非零像素排除：

```text
query keypoint -> mask[y, x] == 0 -> 保留为静态候选点
```

如果没有提供 `--query-mask`，则使用全部查询关键点。

## 3. 位姿估计流程

整个流程如下：

```text
新图片
  -> DISK keypoints/descriptors
  -> 检索参考帧
  -> 查询图与参考帧互检匹配
  -> 参考 keypoint index 查 COLMAP point3D_id
  -> 得到 query 2D point <-> scene 3D point 对应
  -> RANSAC PnP + calibrated focal prior
  -> 新图的 R、t、camera center
```

### 3.1 查询特征

查询图的 DISK 特征预先保存在：

```text
outputs/Friends-monica_apartment/new_image_localization/query_h5/keypoints.h5
outputs/Friends-monica_apartment/new_image_localization/query_h5/descriptors.h5
```

脚本根据 `--query-key` 读取对应 H5 数据集，例如 `query_0001`。如果指定 mask，先通过关键点坐标过滤动态区域。

### 3.2 参考帧检索

`retrieval_candidates.tsv` 提供与查询图外观相似的已注册参考图片。脚本默认读取前 `--top-k` 个候选，并可以通过 `--exclude-reference` 排除某些图片。

排除参考图片很重要。例如，如果查询图片实际是从某个数据集帧复制出来的，为了测试“新图片定位”流程，可以排除这个同名或同源帧，避免直接使用它的相机信息。

### 3.3 DISK 描述子互检匹配

对于每一张参考帧：

1. 对查询描述子和参考描述子做 L2 归一化。
2. 使用描述子点积计算相似度。
3. 查询图到参考图做双近邻 ratio test。
4. 参考图到查询图也做同样的 ratio test。
5. 只保留满足 mutual nearest 和 ratio 条件的匹配。

当前默认参数：

```text
ratio = 0.90
```

代码中的距离是归一化描述子的平方 L2 距离：

```text
d = 2 - 2 * dot(query_descriptor, reference_descriptor)
```

ratio test 使用最佳距离与第二佳距离的比值。互检的目的，是减少重复纹理、人物衣服和相似家具带来的错误匹配。

### 3.4 从 COLMAP track 得到 2D-3D 对应

参考图片已经在 COLMAP 中注册。对参考图片中的每一个匹配 keypoint，脚本读取：

```text
image.point3D_ids[reference_keypoint_index]
```

如果该值为 `-1`，说明这个 keypoint 没有对应的 3D track，会被丢弃。对于有效的 `point3D_id`，再从 `points3D.bin` 读取：

```text
points3d[point3D_id].xyz
```

于是得到：

```text
query pixel (u, v) <-> COLMAP world point (X, Y, Z)
```

多个参考帧的对应会合并，并按 `(query_keypoint_id, point3D_id)` 去重。最终不会直接拿最近参考帧的相机位姿作为答案，而是累积跨帧的 2D-3D 约束后统一求解。

### 3.5 RANSAC PnP 和焦距搜索

新图片的内参通常未知，所以脚本对焦距做一个小范围搜索。对每个候选倍率：

```text
focal = multiplier * max(image_width, image_height)
cx = image_width / 2
cy = image_height / 2
```

当前默认倍率为：

```text
0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 1.9, 2.1
```
如果已知新图来自同一拍摄机位，推荐使用像素焦距先验：

```text
--focal-pixels 2558.567097579417
```

该参数直接覆盖倍率网格，避免粗略倍率把相机中心推偏。焦距可以来自同机位已注册帧的 COLMAP 相机、镜头标定或 EXIF；不要把查询帧的真值位姿写入流程。脚本默认仍保留倍率搜索，便于没有先验时运行。


每个候选内参都运行：

```text
cv2.solvePnPRansac
  iterationsCount = 5000
  reprojectionError = 8 pixels
  confidence = 0.999
  method = EPNP
```

RANSAC 得到内点后，使用 `cv2.solvePnPRefineLM` 做 Levenberg-Marquardt 精化，并重新计算内点重投影 RMSE。所有焦距候选中，优先选择：

1. 内点数最多的结果；
2. 内点数相同则选择 RMSE 更小的结果。

## 4. 位姿定义和公式

OpenCV PnP 输出的是 world-to-camera 变换：

```text
X_camera = R * X_world + t
```

JSON 中保存：

```text
world_to_camera_rotation   = R
world_to_camera_translation = t
```

相机中心通过下面的公式转换到 COLMAP 世界坐标：

```text
C_world = -R^T * t
```

这就是 JSON 中的：
