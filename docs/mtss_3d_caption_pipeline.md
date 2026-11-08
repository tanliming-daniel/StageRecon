# MTSS-3D 视频标注管线

本管线把一个已经切好的 StageRecon clip run 转成可校验、可人工复核、可直接汇总为训练 JSONL 的 `mtss_3d_v1` 数据包。

设计严格保留 MTSS 的四条语义信息流：

- `Reference`：稳定的实体 ID、外观锚点和身份属性。
- `Shot`：镜头区间、摄影语言、可见人物和画面内动作描述。
- `Event`：有音频或可靠逐字稿证据的对白、音效和音乐事件。
- `Global`：场景、整体视觉风格和全局声音环境。

StageRecon 特有的信息位于 `extensions.mtss_3d`，不混入自然语言 Shot：

- `entity_bindings`：SAM3 track 到 `PERSON_N` / Friends 角色身份的绑定。
- `blocking_motion`：人物根节点、初始位置和移动路线约束。
- `geometry`：相机、演员轨迹、帧时间轴和 render job 的文件引用与哈希。
- `evidence`：`observed`、`derived`、`inferred` 的证据链。

## 训练控制协议

训练 prompt 固定使用 `previs_control_v1`。预演视频只控制：

- 构图、粗略场景布置、人物初始位置。
- 人物根节点移动路线和画面内相对位置。
- 景别、相机轨迹和整体时间节奏。

预演视频不控制代理人物外观、局部肢体动作、姿势、表情或僵硬表演。人物身份和外观来自 Reference Bank，动作和表演语义来自 MTSS。协议文本位于 `prompts/training/previs_control_v1.txt`，每条训练记录同时保存协议 SHA-256，避免批量生产中静默漂移。

## 安装

```bash
pip install -r requirements-caption.txt
```

默认身份后端是保守的 OpenCV 颜色直方图，只用于验证整条工程链路。生产身份匹配建议安装 `insightface` 与合适的 `onnxruntime`，并使用 `--identity-backend insightface`。

## 1. 准备 Friends 参考图库

图库按角色目录组织，目录名应对应配置中的 `canonical_identity`：

```text
/data/friends_gallery/
  monica_geller/
    001.jpg
    002.jpg
  rachel_green/
  phoebe_buffay/
  chandler_bing/
  joey_tribbiani/
  ross_geller/
```

生成带实际图片路径的 registry：

```bash
python tools/build_friend_reference_registry.py \
  --gallery /data/friends_gallery \
  --output configs/characters/friends.local.json
```

身份绑定策略：

1. 从 SAM3 稳定 track 中选择多个高质量帧。
2. 使用原图 RGB 和 mask 生成透明背景 RGBA 人物 crop。
3. 对多张 crop 与角色图库做聚合匹配。
4. 只有最佳分数和与第二名的 margin 同时过阈值才绑定角色。
5. 低置信度时回退为匿名 `PERSON_N`，Reference 外观锚点使用当前 track 的 RGBA crop。
6. 同一镜头不允许两个 track 自动绑定到同一 canonical identity。

可先单独查看身份结果：

```bash
python tools/match_character_identities.py \
  --clip-run outputs/scene/clip_runs/clip_id \
  --registry configs/characters/friends.local.json \
  --output-dir outputs/scene/clip_runs/clip_id/captions/identity_check \
  --backend insightface
```

## 2. 单片 dry-run

dry-run 不调用 Qwen，会生成占位语义，但完整执行 manifest、抽帧、人物 crop、身份回退、geometry 打包、Schema 校验、训练记录和 HTML review。

```bash
python tools/run_mtss3d_caption.py \
  --clip-run outputs/Friends-monica_apartment/clip_runs/1028_1791454112 \
  --previs-video outputs/Friends-monica_apartment/clip_runs/1028_1791454112/camera_track_playblast.mp4 \
  --registry configs/characters/friends.local.json \
  --dry-run \
  --no-resume
```

默认输出到 `<clip-run>/captions/mtss_3d_v1/`。关键产物：

- `mtss_3d.json`：最终四流 MTSS + `extensions.mtss_3d`。
- `identity/bindings.json`：track、角色匹配、候选分数和 fallback 结果。
- `geometry/`：相机、演员轨迹、帧时间轴和 render job 的冻结副本。
- `semantic/`：Qwen proposal 或 dry-run proposal。
- `raw/`：发送给模型的请求文本、响应和重试元数据。
- `validation_report.json`：Schema、ID、时间、文件和哈希检查。
- `training_record.json`：下游训练可直接消费的固定 prompt。
- `review.html`：人物锚点、镜头描述、事件和校验结果的人工复核页面。

## 3. 调用 Qwen

默认主路径使用 `qwen3.8-omni-flash`，每个 clip 只进行一次联合标注调用。Qwen 的输入严格限定为：真实 `target_video`（含原始音轨）、从真实 target 中提取的人物参考 crop/Mask、`ref_id` 对照表、target 的真实时长和 FPS。预演视频/playblast 不进入 Qwen 请求，也不进入 Qwen 标注 Prompt；它只作为后续训练样本的控制视频。

```text
真实目标视频（含原始音轨） + 人物参考图/Mask + ref_id + 时长/FPS
                              ↓
                  Qwen3.8-Omni 联合标注
                              ↓
                 Reference + Shot + Event + Global
```

```bash
export DASHSCOPE_API_KEY=...

python tools/run_mtss3d_caption.py \
  --clip-run outputs/Friends-monica_apartment/clip_runs/1028_1791454112 \
  --previs-video outputs/Friends-monica_apartment/clip_runs/1028_1791454112/camera_track_playblast.mp4 \
  --target-video data/clips/1028_1791454112.mp4 \
  --registry configs/characters/friends.local.json \
  --identity-backend insightface \
  --no-resume
```

Omni 直接负责动作和镜头描述、对白逐字转录、对白时间范围、基于画面/声音/嘴部活动的说话人识别、说话人与 `PERSON_*` 的绑定、环境声、音效、音乐和音画对应关系。相机矩阵、角色根节点轨迹和其他 geometry 仍由 StageRecon 真值文件提供，模型不得猜测。

程序从真实 target MP4 的视频流读取 FPS、帧数和时长，并拒绝未知 `ref_id`/speaker、越界或重叠镜头、未关联事件、无逐字文本的对白和非法 JSON。模型返回字段别名时会先做确定性的 Schema 归一化，再校验并重试。

旧的“视觉标注 + 外部 ASR transcript”只作为显式回退；传入 `--transcript`、`--visual-proposal` 或 `--audio-proposal` 时才启用：

```bash
python tools/run_mtss3d_caption.py ... \
  --audio /data/audio/clip.wav \
  --transcript /data/transcripts/clip.txt
```

默认 Omni 路径把真实 target MP4 登记为原始音轨来源。静音片段应返回空 Event Stream，Global Audio 保持 null。

## 4. 人工覆盖身份

误匹配或低置信度样本可以传入人工绑定 JSON：

```json
{
  "bindings": [
    {"track_id": 0, "canonical_identity": "monica_geller"},
    {"track_id": 1, "canonical_identity": "chandler_bing"}
  ]
}
```

```bash
python tools/run_mtss3d_caption.py ... \
  --manual-bindings manual_bindings.json \
  --no-resume
```

## 5. 批量运行

JSONL 每行一个样本：

```json
{"sample_id":"clip_0001","clip_run":"/abs/path/clip_run","previs_video":"/abs/path/playblast.mp4","transcript":"/abs/path/transcript.txt"}
```

```bash
python tools/run_mtss3d_batch.py \
  --manifest manifests/caption_jobs.jsonl \
  --output-root outputs/captions/mtss_3d_v1 \
  --registry configs/characters/friends.local.json \
  --identity-backend insightface \
  --model YOUR_QWEN_VL_MODEL_ID \
  --max-workers 4
```

批处理支持 resume。每条样本独立落盘，失败不会污染其他样本，并在 manifest 旁生成 `*.caption_report.json`。

## 6. 校验与训练集导出

```bash
python tools/validate_mtss3d.py \
  --input outputs/captions/mtss_3d_v1/clip_0001/mtss_3d.json

python tools/build_caption_review.py \
  --input outputs/captions/mtss_3d_v1/clip_0001/mtss_3d.json

python tools/export_mtss3d_training_jsonl.py \
  --input-root outputs/captions/mtss_3d_v1 \
  --output datasets/mtss_3d_v1/train.jsonl \
  --require-complete
```

`--require-complete` 会排除 `needs_review` 和 `dry_run_placeholder`，避免匿名人物或未审样本进入正式训练集。

## 7. 推荐的首轮质量门槛

- 20 段：验证 Schema、路径、ID 和 geometry 引用。
- 100 段：统计身份混淆、动作漏标、时间偏差、模型 JSON 重试率。
- 只有 `validation_report.json` 通过且人工审阅完成的样本才标记为 `complete`。
- Schema、prompt 和身份 registry 版本冻结后再进入 500–1,000 段扩展。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q stagerecon tools tests
```
