# StageRecon 场景更新与跨场景接入

本流程在 H100 的 `/root/userdata/300workspace/StageRecon` 运行。项目的生产入口是 `tools/run_clip_render_data_batch.py`。前两阶段产出相机和人物数据，Blender 单独负责导入、验收和渲染。本文命令在项目根目录执行。

## 当前发布版本

| 项目 | 当前值 |
|---|---|
| 基础场景 | `data/scene_with_clip_animation_366923a7.blend` |
| 场景 SHA-256 | `366923a76853bd956b0db8ed5e829245c6c998750b600834345d569ae135a457` |
| 活动配置 | `/root/userdata/300workspace/merg3r_work/omega_20261119_reviewed172_ba_regularized/batch_scene.json` |
| 仓库中的配置副本 | `configs/scenes/Friends-monica_apartment.omega_batch.json` |
| 家具标注、渲染显示名单 | `configs/scenes/Friends-monica_apartment.omega_layout.json` |
| 已验收的接触几何 | `configs/scenes/omega_seat_contacts_366923a7.json` |
| 24 镜头及播放顺序 | `configs/scenes/Friends-monica_apartment.review24.json` |
| 审定数据 | `data/scene_releases/Friends-monica_apartment/366923a7_v1/` |

### 从主入口出片

```bash
cd /root/userdata/300workspace/StageRecon
PY=/root/userdata/300workspace/da3/bin/python
BL=/root/userdata/300workspace/blender-5.1.1-linux-x64/blender
BASE=data/scene_with_clip_animation_366923a7.blend
CLIPS=data/clips/Friends_S08_PublicHD_OmniShotCut_v1_5
CASES=configs/scenes/Friends-monica_apartment.review24.json
OUT=outputs/Friends-monica_apartment/clip_render_data_s08_omega_shoulders_p075_366923a7

"$PY" tools/run_clip_render_data_batch.py --clips-root "$CLIPS" --case-list "$CASES" --output-root "$OUT"
CUDA_VISIBLE_DEVICES=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 LP_NUM_THREADS=3 \
  "$PY" tools/render_omega_comparison.py --clips-root "$CLIPS" --case-list "$CASES" \
  --bundles "$OUT" --base-blend "$BASE" --blender "$BL" --jobs 4 \
  --output outputs/Friends-monica_apartment/GT_Blender_24.mp4
```

左侧 GT 与右侧渲染按帧配对。合辑按清单顺序编码，校验每段帧数和最终视频解码帧数。本批应为 24 段、1,387 帧、24000/1001 fps。检查背面 0146、正面 0116，以及第 19 段 S08E22 · 0025。新镜头用 `--episode S08E22` 或 `--include <文件名片段>` 选择；不传 `--case-list` 就按视频目录选择。预览执行计划可加 `--dry-run`。

只导入一个镜头时，把输出写到数据包目录外。数据包应始终只有 `frames.json`、`camera_track.json`、`actor_tracks.json`、`render_job.json` 四个文件。

```bash
"$PY" tools/run_clip_render_job.py \
  --job "$OUT/S08E22/general_0025_global_0110_f00006685-f00006740/render_job.json" \
  --base-blend "$BASE" --blender-executable "$BL" \
  --output-blend outputs/review_19/scene.blend --blender-metadata outputs/review_19/import.json
```

### 已审定镜头与新镜头

主入口按 clip 相对路径和视频字节 SHA-256 查找审定包，同时校验场景、点云、Omega 预测文件、场景参数签名、渲染任务摘要和三份数据摘要。匹配时直接采用已验收的相机和轨迹，manifest 记录 `published_review`。这些资产已独立于实验目录。新视频走 SAM 3 → Omega 定位与 DA3 → SAM 3D Body → 场景约束。要研究同一视频从头推理的差异，使用 `--no-use-reviewed` 和新输出目录。弱标定镜头可能无法从头通过定位质量检查，审定包保留原有相机来源记录。

人物实例分数门槛为 **≥ 0.75**。角色确认后在整个 clip 内保留；双肩支持位置，头部旋转驱动眼线。Blender 按数值 track_id 固定配色，过滤角色、切换镜头或导入带旧 color_slot 的审定包均使用相同映射。ID 0–9 依次为红、天蓝、绿、金黄、紫、青绿、粉、棕、黄绿、靛蓝，其他 ID 按数值计算色相。track_id 是镜头内的跟踪编号，同色不代表跨镜头是同一位演员。新推理只对可靠双肩覆盖充分、人体深度与代理投影宽度都明显异常的轨迹重新估深，再利用配置中可见的静态网格射线交点限制躯干深度。GPU 4 的共享显存不足时，失败会写入 manifest，可在资源释放后用相同命令重试。

第 19 段的右下角相框是人工核实的图像区域。配置中的 `reviewed_image_regions` 绑定视频摘要；矩形坐标为原始图像的归一化 `[左, 上, 右, 下]`。只有同一轨迹的全部观测框都落在区域内才排除。其他镜头里的电视、照片需要独立证据；不要复制这段的屏幕坐标或角色 ID。

## 更新同一场景

1. **在 Blender 检查对齐与单位。** 活动场景中保留名为 `pointcloud` 的对齐对象。其 `matrix_world` 把 Omega 世界坐标映射到 Blender。当前求解假设 Blender 的 +Z 朝上，1 个 Blender 世界单位代表 1 米；点云对象使用正的均匀缩放。用门宽、沙发长等已知尺寸校对实际米制尺度。保存时保留点云对象的对齐变换，避免把变换烘焙进网格后丢失矩阵。
2. **检查最终渲染资产。** 用相机渲染抽查门、门框、窗边家具、坐垫和靠背。查对象及父集合的相机开关、活动 View Layer 的排除状态、纹理路径。外部贴图可打包到 blend，或放在 H100 可解析的相对目录。已有的 `ClipActors`、`ClipCamera` 属于生成内容，导入器会清理；布景资产放到单独集合。基础文件按版本保存到 `data/`。
3. **更新 layout。** 展示必须入镜、但文件里关闭了对象渲染的物体，填写 `render_objects` 的准确对象名。该设置只打开对象级开关，父集合仍需允许渲染。`hide_collections` 填需要在导入时隐藏的旧相机和人物集合。家具移动后，需重新量取可坐线段端点、前向和座面高度。
4. **生成候选配置，先在新输出目录验收。** 下方命令从当前场景的实际网格重新采样。候选文件可以放在另一个目录，原活动配置继续可用。

```bash
R=/root/userdata/300workspace/merg3r_work/omega_20261119_reviewed172_ba_regularized
"$BL" -b "$BASE" --python-exit-code 2 -P tools/prepare_omega_scene.py -- \
  --layout configs/scenes/Friends-monica_apartment.omega_layout.json \
  --reconstruction "$R" --pointcloud "$R/points_static_p70_no_sampling.ply" \
  --reference-images data/sparse_reconstruction_and_image_data/Friends-monica_apartment/images \
  --output "$R/candidate/batch_scene.json"
```

该步骤生成 `batch_scene.json` 和带摘要的接触几何 JSON，并记录当前 blend、PLY、相似变换、世界上方向、每米 Omega 单位数、渲染对象名和 Blender 可执行路径。它读取 blend，不会另存或覆盖源场景。把候选配置作为批处理的 `--omega-scene-config "$R/candidate/batch_scene.json"`，并指定新的 `--output-root` 测试。`--omega-reconstruction` 继续指向存放 `predictions.npz` 的重建目录。验收后备份旧配置，将候选 JSON 原子替换为 `$R/batch_scene.json`，保留其中引用的接触几何文件。当前正式版本继续使用已验收的接触网格。

5. **更新审定清单。** 新场景会产生新的 SHA。重新检查人物、相机、家具关系后再发布。对完全相同的基础文件调整座位、显示规则或审定区域，也应使用新输出目录和新的发布目录。已有审定索引会校验场景参数签名。

```bash
"$PY" tools/omega_review.py --scene "$R/candidate/batch_scene.json" \
  --cases "$CASES" --clips-root "$CLIPS" --bundles "$VERIFIED_BUNDLES" \
  --release "$NEW_RELEASE_DIR"
```

`VERIFIED_BUNDLES` 指验收后的四文件包根目录，`NEW_RELEASE_DIR` 是本次发布的目录。将生成的 `index.json` 路径填入候选配置的 `review_index`，同步更新项目内配置副本。没有审定结果的新场景可直接运行推理。

## 接入其他场景

| 需要准备 | 具体要求 |
|---|---|
| 重建与参考图 | 同一 Omega 坐标系的 PLY、参考图、`predictions.npz`。定位器读取 `names`、`depth`、`static_mask`、`intrinsics`、`extrinsics`。图名与实际参考图文件对应，静态掩码要尽量排除人物。平移、深度和点云尺度须一致。 |
| Blender 布景 | 用新场景自身的至少几个分散静态参照对齐平移、旋转与统一尺度；用另一视角检验。导入器使用 `pointcloud` 对象的完整世界矩阵。资产保留清楚、稳定的对象名。 |
| 地面 | `floors` 填可站立的近似水平地面网格。当前地面表使用一块网格一个高度和逐行覆盖范围，多层地板、楼梯或有大空洞的地面应按平坦连通区域拆网格。 |
| 座位 | `seats` 以座位标签为键。每项含 `object`、世界空间三维 `left/right`、二维 `front_direction` 和世界 Z `height`。左右端点沿坐垫中央可坐带取，前向从靠背指向坐垫前缘。导入前后检查前向符号。多个座面可以分开标；`backrest_seat` 可指定其中一个使用当前专门的靠背遮挡修正，默认键为 `sofa`。 |
| 遮挡 | `occluders` 填参与座位和遮挡判定的家具网格。`depth_mesh_objects` 填异常近景重新测深时允许射线命中的静态 mesh 对象；应使用实际参与渲染的墙、家具、门等，跳过点云、人物、辅助体及隐藏集合。 |
| 显示与复核 | `render_objects`、`hide_collections` 使用新场景自己的名称。另建回归镜头清单，覆盖正反打、近景、多人坐姿、出入画，以及电视或照片误检。 |

可以从精简 layout 开始，再逐项添加家具标注：

```json
{
  "render_objects": [],
  "hide_collections": [],
  "seats": {},
  "floors": [],
  "occluders": [],
  "depth_mesh_objects": []
}
```

精简配置可先检验相机和人体投影。加入地面、座位与靠背后，再验收坐姿和遮挡。鱼眼、镜面反射、曲面或运动座椅，以及一段视频内的硬切，都需要额外处理；当前主线按单镜头、固定布景和针孔相机工作。跨场景运行时显式设置 `--omega-reconstruction`、`--omega-images`、`--clips-root` 和 `--output-root`。若 Blender 可执行路径不同，用 `STAGERECON_BLENDER` 指定异常深度复核所用的程序，对照视频命令使用 `--blender`。

## 验收与回看

- **空场景**：比较分散的固定物，例如门框、靠背上沿、窗框和桌角。正反两个视角都检查；多个物体同方向偏移时，优先检查相机和点云变换。
- **人物**：确认肩线和画面中身体所占宽度；个别头过大时，同时检查相机深度与代理的实际尺寸。用原片露出的头部面积检查靠背遮挡。
- **数据与视频**：每个确认角色在整段只用一个对象，无零缩放，渲染可见性开启，不同角色颜色互异。比较视频还会检查 GT 与渲染帧数相同。自动通过后仍需抽看构图和自然程度。
- **复用边界**：更新源视频或重建文件时重新定位；替换 blend 或变更接触几何时重新验收。保留原始审定包、对应的 blend 与配置，便于回溯。
