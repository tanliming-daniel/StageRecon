# Friends 分集人物候选图库

`tools/export_episode_character_gallery.py` 从 Monica 公寓稀疏重建使用的图片库自动建立人物候选图库。当前约定是：**每集、每个人物候选组只保留 3 张图**。输出用于后续人工筛选和生图 API 生成正式 Reference Bank，不直接作为最终身份图库。脚本不读取或解码 MKV，也不做跨集身份关联。

## 默认输入与输出

- 输入：`data/sparse_reconstruction_and_image_data/Friends-monica_apartment/images`
- 输出：`data/raw/friends_gallery`

目录结构：

```text
data/raw/friends_gallery/
  S08E05/
    person_001/
      01.png
      02.png
      03.png
    person_002/
      01.png
      02.png
      03.png
    contact_sheet.jpg
    manifest.json
```

人物在正式身份图库尚未建立时以 `person_001`、`person_002` 编号；编号只在本集内有效。同一人物被拆成多个编号是允许的，后续人工合并即可。`contact_sheet.jpg` 用于快速人工确认，`manifest.json` 记录每张图的源图片、帧号、时间戳、人脸框、SAM3 分数、人体完整度和抠图后端。

## 处理逻辑

1. 按 `Friends_S08E05_00012827.jpg` 中的 `S08E05` 自动分集，并读取末尾源帧号。
2. 每集最多均匀扫描 160 张重建关键图，用 OpenCV 正脸 + 双眼校验寻找可靠脸框。
3. 用脸部特征为主、服装颜色为辅做本集内保守聚类；不同集之间不关联。
4. 同一帧同时出现的两张脸禁止合并为同一人物。
5. 每组最多选 6 张偏远景候选，交给常驻的 SAM3 `person` worker 定位人物 mask。
6. 基于 SAM3 人物前景的躯干颜色过滤混入的其他服装或人物。
7. 按人物 mask 高度、脸高比例、脸以下身体长度和画面边缘截断情况排序，优先保留身体最完整的 3 张。
8. mask 只作为横向先验：最终裁剪固定保留原帧完整高度 `0:H`，宽度使用人物 mask 的 `x_min:x_max`；输出普通 RGB PNG，保留原背景，不输出 mask 或透明通道。

由于源关键图多数是中近景，脚本只能在现有画面中“尽量”选择全身或大半身，不能补出源图未拍到的腿部。严格判定结果写在 `manifest.json` 的 `full_body_likely` 和 `completeness_score` 字段中。

## 运行

先处理单集：

```bash
python tools/export_episode_character_gallery.py \
  --episode S08E05
```

处理整季：

```bash
python tools/export_episode_character_gallery.py
```

默认由系统 Python 完成人脸检测/聚类，并自动调用 `/root/userdata/300workspace/sm3r/bin/python` 运行常驻 SAM3 worker。脚本可续跑；完整且参数一致的集会自动跳过。参数变化后重建某集：

```bash
python tools/export_episode_character_gallery.py \
  --episode S08E05 \
  --overwrite
```

快速预览聚类、不加载 SAM3，并直接从原帧裁剪 RGB 人物候选：

```bash
python tools/export_episode_character_gallery.py \
  --episode S08E05 \
  --no-sam3
```

`--no-sam3` 不生成 mask 或透明通道。它以人脸为中心裁剪原始 RGB 图像，顶部保留头发余量，底部延伸到源图底边，优先避免把腿脚裁掉；结果会保留人物周围的原始场景，适合粗筛后再人工挑选。

## 常用调节

- `--min-observations 2`：保留只检测到两次的客串角色。
- `--max-characters 20`：提高每集允许的角色上限。
- `--assign-threshold` / `--merge-threshold`：数值越高越不容易把不同人物误合并，但同一人物可能被拆成多个编号。
- `--min-face-pixels 30`：允许更小的远景脸，可能找到更多全身镜头，但误检会增加。
- `--candidate-images-per-person 9`：每组送入 SAM3 的候选数越多，越可能找到更完整的身体，但运行更慢。
- `--max-source-images 0`：扫描该集全部重建图；默认 160 张均匀采样。
- `--sam3-python /path/to/python`：指定包含 SAM3 依赖的 Python 环境。

默认策略是“宁可同一人物拆成多个编号，也不要不同人物混在一组”。人工筛选时先看每集的 `contact_sheet.jpg`，删除误组并合并重复 `person_###`；正式身份匹配放在后续阶段完成。

## 当前批次

- 已处理 20 集：S08E02–S08E10、S08E12–S08E22。
- 共导出 103 个人物候选组、309 张透明 PNG。
- 每个保留的 `person_###` 都严格包含 3 张图。
- 严格判定为完整全身的源图很少；其余结果已尽量选择大半身而不是近景头像。
