# 新图片到 Blender 预览图

本文定义 Friends Monica apartment 当前可运行的数据生成链路：给定一张不属于
COLMAP 训练集的新剧集帧，在既有场景坐标系中预测它的相机位姿，并从已实例化的
Blender 场景以该相机导出原生视口预览图。

```text
new image
  -> SAM3 person/dynamic foreground mask
  -> DISK local features
  -> registered-frame retrieval
  -> DISK 2D matches + COLMAP tracks
  -> RANSAC PnP pose in COLMAP world
  -> pointcloud transform converts pose to Blender
  -> PredictedCamera_query_0001
  -> Solid + Texture + Studio OpenGL viewport PNG
```

这是定位和预览链路，不会把查询图片加入 COLMAP，不会重新建图，也不会保存新的
`.blend` 文件。

## 场景和依赖

| 项目 | 当前路径/作用 |
| --- | --- |
| COLMAP 场景 | `sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/colmap` |
| 参考图片和 DISK H5 | 同目录的 `images/` 和 `h5/` |
| Blender 场景 | `sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/scene_with_predicted_camera.blend` |
| 对齐物体 | Blender 中的 `pointcloud`，其 `matrix_world` 定义 COLMAP 到 Blender 的对齐 |
| DISK 权重 | `outputs/Friends-monica_apartment/new_image_localization/models/depth-save.pth` |
| DISK 全局检索缓存 | `outputs/Friends-monica_apartment/models/disk_global_retrieval_index.npz` |
| Python 环境 | `/root/userdata/300workspace/sm3r/bin/python` |
| SAM3 源码和权重 | `/root/userdata/300workspace/sam3` 和 `/root/userdata/300workspace/ckpt/sam3/sam3.pt` |
| GPU Blender | `/root/userdata/300workspace/zdj/blender/blender` |
| OpenGL Display | `:88` |

预览使用 Blender 中已经实例化的场景资产，不是渲染 `pc.ply` 点云。

## 单张图片运行

在仓库根目录运行：

```bash
/root/userdata/300workspace/sm3r/bin/python \
  tools/run_new_image_to_blender.py \
  --query-image /absolute/path/to/new_frame.jpg \
  --blender-input sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/scene_with_predicted_camera.blend \
  --output-dir outputs/Friends-monica_apartment/new_image_runs/my_frame \
  --blender-executable /root/userdata/300workspace/zdj/blender/blender \
  --python-executable /root/userdata/300workspace/sm3r/bin/python \
  --viewport-display :88 \
  --camera-name PredictedCamera_query_0001 \
  --preview
```

查询图会原样复制为 `query_images/query.<原始扩展名>`。DISK 只将网络输入尺寸向上
补齐到 16 的整数倍；PnP 像素坐标仍对应查询图实际尺寸。

如果有可信像素焦距，可缩小内参不确定性：

```bash
  --focal-pixels 2558.5671
```

默认会先用本地 SAM3 以文本提示词 `person` 提取人物 mask，并把人物区域向外扩展
12 px。推理只使用本地源码、权重和 `sm3r` 环境；GPU 模型执行约 22 秒，独立进程首次
从网络盘加载源码和权重时会额外花费几十秒。非零像素不会进入检索描述子、2D 匹配或
PnP 对应，输入图本身不会被改写。

如已有更可信的人工或视频跟踪 mask，`--query-mask` 优先于 SAM3 自动结果，非零区域会
被排除：

```bash
  --query-mask /absolute/path/to/dynamic_mask.png
```

若需要比较未剔除人物的旧逻辑，可显式关闭自动 mask：

```bash
  --person-mask off
```

每个自动 mask 样本还会保存 `person_mask_sam3.png`、`person_mask_sam3_overlay.jpg` 和
`person_mask_sam3.json`，用于核对人物边界、检测分数和被排除面积。

## 批量运行

`tools/run_image_batch_to_blender.py` 递归读取 `jpg/jpeg/png/webp/bmp`，每张图独立
生成一套输出。`--resume` 默认开启，会跳过在 `batch_manifest.json` 中标记为成功的
样本；此前失败或中断的目录会原地刷新。

```bash
/root/userdata/300workspace/sm3r/bin/python \
  tools/run_image_batch_to_blender.py \
  --input-dir data \
  --output-root outputs/Friends-monica_apartment/data_batch_YYYYMMDD \
  --blender-input sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment/scene_with_predicted_camera.blend \
  --blender-executable /root/userdata/300workspace/zdj/blender/blender \
  --python-executable /root/userdata/300workspace/sm3r/bin/python \
  --viewport-display :88 \
  --camera-name PredictedCamera_query_0001 \
  --resume
```

批量清单位于：

```text
outputs/Friends-monica_apartment/data_batch_YYYYMMDD/batch_manifest.json
```

每项记录 `status`、相机中心、PnP 内点数、RMSE、相机 JSON 和预览图路径。定位
失败的图片保持 `failed`，不会伪造相机或预览。

`status: ok` 只表示流程完整执行。每个成功项还带有 `pose_quality`：`accepted` 需要至少
40 个 PnP 内点且 RMSE 不高于 4.5 px，才会标记 `dataset_eligible: true`；20 个内点和
5 px 以内为 `review`，必须人工核对预览；其余为 `rejected`。这层标签不会阻止导出预览，
因为预览正是人工判断 `review` 样本的依据。

## 相机定位

### 1. SAM3 动态前景剔除

新图先由 `tools/extract_person_mask_sam3.py` 调用本地 SAM3 图像模型。它对 `person`
提示的所有高于置信度阈值的实例求并集，并以形态学膨胀覆盖头发、衣物边缘等不稳定
特征。mask 中 `255` 表示动态区，`0` 表示可用于静态场景定位的区域。

参考帧若存在原始 Sitcoms3D `segmentations/thing/<frame>.png`，则同样过滤类别值 `1`
（person）的 DISK 点。这个数据集中该目录只覆盖部分帧，因此缺失 mask 的参考帧会被
明确记录为未覆盖，而不是错误地当作全背景。

### 2. DISK 查询特征

`external/disk/detect.py` 用与参考 H5 一致的 `depth-save.pth` 输出：

```text
query_h5/keypoints.h5
query_h5/descriptors.h5
```

查询和参考必须使用同一 DISK 模型和描述子格式，否则参考 keypoint 将无法可靠地
映射回 COLMAP track。

### 3. 两级已注册帧检索

第一层快速计算 YCrCb 颜色和边缘外观距离，选择前 20 张注册参考帧，适合包含
较多静态布景的普通镜头。

人物近景或大面积遮挡时，第一层可能没有足够静态点。此时 runner 自动执行回退：

1. 仅以新图 static mask 外的 DISK 描述子做归一化全局平均池化；
2. 在 `disk_global_retrieval_index.npz` 中检索前 200 个候选；
3. 只保留 `images.bin` 内已注册的参考帧；
4. 以较宽松的 `ratio=0.95` 重跑局部匹配和 PnP。

回退较慢，但能找回分散在门、墙、橱柜、台面等区域的静态特征。

### 4. 2D-3D 对应和 PnP

每个候选帧中，查询和参考特征经过 mutual nearest-neighbor ratio test，再以基础
矩阵 RANSAC 去除不符合两视图几何的匹配。参考点通过：

```text
COLMAP image.point3D_ids[reference_keypoint_index]
```

映射成：

```text
query pixel (u, v) <-> COLMAP world point (X, Y, Z)
```

同一查询点若对应多个 3D 点，只保留跨帧支持数最多的一个。随后对焦距候选运行
`cv2.solvePnPRansac`，以 LM 精化，选择内点数最多且重投影 RMSE 最小的结果。

实用质量基线是至少 20 个 PnP 内点、inlier RMSE 不高于约 5 px。它不是绝对保证；
仍应把预览与输入中的静态门、墙、橱柜、桌椅等结构比对。

### 5. COLMAP 到 Blender

PnP 结果为：

```text
X_camera = R * X_colmap + t
C_colmap = -R^T * t
```

`export_predicted_camera_info.py` 读取 `pointcloud.matrix_world`，将相机位姿对齐到
Blender，并以 `lens_mm = fx * 36 / image_width` 写出 `camera_info.json`。该脚本不
修改也不保存输入场景。

## 原生 Blender 预览

`capture_viewport_preview.py` 在独立 Blender 进程中：

1. 将 `camera_info.json` 应用到 `PredictedCamera_query_0001`；
2. 同时将它设为场景 active camera 和 `VIEW_3D` camera；
3. 临时隐藏 `actor1`、`camera` collection；
4. 创建不保存的临时 preview scene，复用原场景集合；
5. 使用 Workbench `Solid + Texture + Studio`，关闭 shadows、cavity、specular；
6. 通过 `bpy.ops.render.opengl(write_still=True)` 导出 960 x 540 PNG。

它不是 UI 截屏，也不是 Cycles/Eevee 光照渲染。预览仅显示静态场景资产，输入视频
中的演员不会出现。`viewport_capture.json` 必须记录：

```json
{
  "active_camera": "PredictedCamera_query_0001",
  "viewport_camera": "PredictedCamera_query_0001",
  "viewport_color_type": "TEXTURE",
  "render_color_type": "TEXTURE"
}
```

这四项用于确认预览没有误用旧相机或 Attribute 颜色。

## 每张图的产物

```text
<output-dir>/
  query_images/query.<ext>             # 原始查询图副本
  person_mask_sam3.png                 # 自动人物动态区，255 为排除
  person_mask_sam3_overlay.jpg         # 红色人物区核对图
  person_mask_sam3.json                # SAM3 分数、框和动态面积
  query_h5/keypoints.h5                # 查询 DISK 特征
  query_h5/descriptors.h5
  retrieval_candidates.tsv             # 快速外观候选
  disk_global_candidates.tsv           # 仅 fallback 时存在
  localization_colmap.json             # COLMAP R/t、内点、RMSE
  camera_info.json                     # 可导入的 Blender 相机信息
  import_predicted_camera.py           # 不保存 blend 的相机导入脚本
  predicted_camera_preview.png         # 原生 OpenGL 预览
  viewport_capture.json                # 相机和材质模式核对信息
  run_manifest.json                    # 产物和质量指标汇总
```

在已有 Blender 场景中导入某项位姿时，只运行该样本生成的
`import_predicted_camera.py` 并传入 `camera_info.json`；脚本只更新
`PredictedCamera_query_0001`，不会执行 `save_as_mainfile`。
