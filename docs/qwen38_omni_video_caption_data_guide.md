# Qwen3.8-Omni 视频 Caption 数据处理说明

本文说明 StageRecon 当前如何使用 Qwen3.8-Omni 处理视频 caption 数据，覆盖输入数据契约、人物身份绑定、模型请求、JSON 归一化与校验、MTSS-3D 合并、批处理、质检、训练集导出和常见故障。

本文以仓库中的实际实现为准，主要对应以下文件：

- tools/run_mtss3d_caption.py：单个 clip 的命令行入口。
- tools/run_mtss3d_batch.py：批量处理入口。
- stagerecon/captioning/pipeline.py：端到端编排。
- stagerecon/captioning/providers/qwen.py：Qwen OpenAI-compatible 请求适配器。
- prompts/mtss_3d/omni_joint_v1.txt：联合音视频标注系统提示词。
- schemas/qwen_omni_mtss_proposal_v1.schema.json：Qwen 原始 proposal 的冻结结构。
- schemas/mtss_3d_v1.schema.json：最终 MTSS-3D 数据结构。
- configs/mtss3d/default.json：默认模型与运行参数。

## 1. 一句话理解整条链路

StageRecon 先从 clip run 中得到稳定人物 track 和人物参考 crop，再把真实 target MP4（含其自身音轨）、人物参考图、允许使用的 PERSON_* ID、视频时长和 FPS 一次性发送给 qwen3.8-omni-flash。模型返回 Reference、Shot、Event、Global 四类语义信息，程序随后做确定性归一化、Schema 校验、交叉引用校验、语义重试，并与 StageRecon 的相机/人物轨迹等几何真值合并成 mtss_3d_v1 数据包。

~~~text
StageRecon clip run
  ├─ 原始帧、SAM3 人物 track 和 mask
  ├─ camera_track / actor_tracks / render_job
  └─ source video
          │
          ├─ 人物 crop 与身份绑定 ──> PERSON_MONICA / PERSON_1 / ...
          ├─ 冻结 geometry 与哈希
          └─ ffprobe target video 的真实时长、FPS、帧数
                              │
                              ▼
       target MP4（使用其内嵌音轨）+ PERSON_* 参考图 + 时间轴
                              │
                              ▼
                    qwen3.8-omni-flash
                              │
                              ▼
          Reference + Shots + Events + Global proposal
                              │
                  归一化、校验、必要时重试
                              │
                              ▼
       mtss_3d.json + training_record.json + review.html
~~~

需要特别记住三个边界：

1. 发送给 Qwen 的是 target_video，不是 previs/playblast。
2. Qwen 使用的是 target_video MP4 自己内嵌的音轨；单独传入的 --audio 不会被拼进 Omni 请求。
3. Qwen 只生成语义标注，不生成或猜测 3D 坐标、相机矩阵、骨骼运动和人物世界坐标轨迹。

## 2. 当前默认路径与旧回退路径

### 2.1 默认：Qwen3.8-Omni 联合路径

满足以下条件时，pipeline 使用 Omni 联合路径：

- 没有显式传入 --transcript；
- 没有显式传入 --visual-proposal；
- 没有显式传入 --audio-proposal；
- 或者显式传入了 --omni-proposal。

默认路径对每个 clip 做一次联合语义任务：

- 看完整 target 视频；
- 使用 target MP4 的原始内嵌音频；
- 识别镜头切换、动作、互动、构图和摄影语言；
- 逐字转录对白；
- 估计对白和声音事件的起止时间；
- 根据画面可见性、嘴部活动和声音连续性，把说话人绑定到允许的 PERSON_*；
- 标注音效、音乐、环境声和音画对应关系。

### 2.2 显式回退：视觉与音频分离路径

只要传入 --transcript、--visual-proposal 或 --audio-proposal 中任意一个，默认 Omni 路径就会关闭，进入旧的分离式路径。

分离式路径中：

- 视觉模型输入为人物参考图和按时间抽取的 observation 帧，不直接使用完整视频；
- 音频模型输入为已经带时间戳的 transcript 文本；
- 如果使用 --transcribe-moss，程序先调用独立 MOSS runtime 生成 transcript，然后自动进入分离式路径；
- audio proposal 和 visual proposal 最后再合并为 MTSS-3D。

这个回退适合以下情况：

- Qwen Omni 后端无法读取视频内音轨；
- target MP4 超过当前 Base64 内联限制；
- 需要用独立 ASR/说话人分离结果替换 Omni 转录；
- 已有人工修订的 visual/audio proposal，希望跳过对应模型调用。

## 3. 输入数据契约

### 3.1 clip-run 目录至少需要什么

--clip-run 指向一个已经完成 StageRecon 前处理的 shot/clip 目录。程序会读取：

~~~text
<clip-run>/
  frames.json                         必需
  frames/                             必需，保存解帧图像
  clip_manifest.json                  可选，但推荐
  masks/person_tracks.json            必需，或由 clip_manifest 指向其他文件
  masks/instances/<track>/<frame>.png 必需，人物实例 mask
  camera_track.json                   必需，或由 clip_manifest 指向其他文件
  actor_tracks.json                   必需，或由 clip_manifest 指向其他文件
  render_job.json                     可选
  camera_track_playblast.mp4          常作为 --previs-video
~~~

clip_manifest.json 或 frames.json 中必须能找到 source video 路径，且该路径在运行机器上真实存在。

### 3.2 三种视频不要混淆

| 名称 | 来源 | 是否发给 Qwen | 用途 |
|---|---|---:|---|
| source_video | clip run 记录的原始视频 | 仅当它也是 target_video 时 | 数据溯源和默认 target |
| target_video | --target-video；未指定时等于 source_video | 是 | Qwen 视觉与音频语义标注的真实输入 |
| previs_video | --previs-video，通常是 Blender playblast | 否 | 训练阶段的粗 3D 控制视频 |

虽然 previs_video 不发送给 Qwen，但它仍是必填参数，因为最终 training_record.json 必须记录控制视频。

### 3.3 时间轴以 target_video 为准

程序使用 ffprobe 读取 target_video 的：

- avg_frame_rate；
- nb_frames，缺失时用 duration × FPS 估算；
- 视频 duration。

最终 MTSS 的 timeline 使用 target_video 时间轴，而不是 frames.json 的时间轴。frames.json 中的 StageRecon 几何时间轴会单独写入 input_manifest.json 的 stage_geometry_timeline，方便检查两者是否一致。

所有语义时间区间使用秒，采用左闭右开约定：

~~~text
[start_sec, end_sec)
~~~

### 3.4 音频输入的真实行为

在 Omni 路径中，模型请求始终发送：

~~~text
manifest.inputs.target_video
~~~

因此：

- --audio 只改变 manifest 和最终 assets.audio 的登记来源；
- --audio-from-source 只把 source video 登记为 audio asset；
- 它们不会把外部 WAV、MP3 或另一段视频音轨附加到 Omni 请求；
- 如果需要 Qwen3.8-Omni 分析一条外部音轨，应先把音轨正确 remux 到 target MP4；
- 如果不希望 remux，应提供 transcript/audio proposal，改走分离式回退路径。

## 4. 环境准备

### 4.1 Python 依赖

在 StageRecon 根目录执行：

~~~bash
pip install -r requirements-caption.txt
~~~

基础依赖包括 NumPy、OpenCV 和 jsonschema。生产身份识别可额外安装 InsightFace 以及与运行设备匹配的 onnxruntime 或 onnxruntime-gpu。

### 4.2 系统依赖

ffprobe 必须在 PATH 中，因为 target 时间轴由 ffprobe 读取：

~~~bash
ffprobe -version
~~~

### 4.3 API Key

默认配置读取环境变量 DASHSCOPE_API_KEY：

~~~bash
export DASHSCOPE_API_KEY=YOUR_KEY
~~~

也可在单条命令中使用 --api-key，但批处理入口没有 --api-key 参数，批处理建议统一使用环境变量。

### 4.4 默认 Qwen 配置

configs/mtss3d/default.json 当前默认值为：

~~~json
{
  "qwen": {
    "base_url": "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    "model": "qwen3.8-omni-flash",
    "api_key_env": "DASHSCOPE_API_KEY",
    "structured_output": "json_object",
    "reasoning_effort": "none",
    "modalities": ["text"],
    "max_tokens": 8192,
    "timeout_sec": 600,
    "max_retries": 2
  }
}
~~~

命令行的 --base-url 和 --model 会覆盖配置文件中的对应值。其他参数需要复制 default.json、修改后通过 --config 指定。

## 5. 单条数据处理

### 5.1 先做 dry-run

dry-run 不调用 Qwen，但仍执行输入读取、人物 crop、身份绑定、geometry 冻结、最终 Schema 校验、训练记录生成和 review 页面生成。

~~~bash
cd /root/userdata/300workspace/StageRecon

python tools/run_mtss3d_caption.py \
  --clip-run outputs/Friends-monica_apartment/clip_runs/1028_1791454112 \
  --previs-video outputs/Friends-monica_apartment/clip_runs/1028_1791454112/camera_track_playblast.mp4 \
  --target-video data/clips/1028_1791454112.mp4 \
  --registry configs/characters/friends.local.json \
  --dry-run \
  --no-resume
~~~

dry-run 的最终 status 为 dry_run_placeholder，不能通过 --require-complete 进入正式训练集。

### 5.2 正式调用 Qwen3.8-Omni

~~~bash
cd /root/userdata/300workspace/StageRecon
export DASHSCOPE_API_KEY=YOUR_KEY

python tools/run_mtss3d_caption.py \
  --clip-run outputs/Friends-monica_apartment/clip_runs/1028_1791454112 \
  --previs-video outputs/Friends-monica_apartment/clip_runs/1028_1791454112/camera_track_playblast.mp4 \
  --target-video data/clips/1028_1791454112.mp4 \
  --registry configs/characters/friends.local.json \
  --identity-backend insightface \
  --identity-device cuda \
  --no-resume
~~~

默认输出目录为：

~~~text
<clip-run>/captions/mtss_3d_v1/
~~~

也可以用 --output-dir 指定其他目录。

### 5.3 不调用 API，使用已有 Omni proposal

调试 merge、validation 或人工修改模型结果时，可以提供预计算 JSON：

~~~bash
python tools/run_mtss3d_caption.py \
  --clip-run /path/to/clip_run \
  --previs-video /path/to/playblast.mp4 \
  --target-video /path/to/target.mp4 \
  --registry /path/to/registry.json \
  --omni-proposal /path/to/qwen_omni_proposal.json \
  --no-resume
~~~

输入仍会经过 normalize_omni_proposal 和 validate_omni_proposal，不是无条件信任。

### 5.4 resume 行为

--resume 默认开启。如果输出目录中已经存在 mtss_3d.json，pipeline 不会重新做身份绑定和模型调用，只会重新生成 validation_report.json 后返回。

以下修改发生后必须使用 --no-resume：

- 换了 target video 或音轨；
- 修改了 prompt、Schema、registry 或人工身份绑定；
- 修改了 Qwen model/base URL/config；
- 想重新请求模型或重新生成人物 crop；
- 修复了旧输出中的错误。

## 6. 人物参考图与稳定 ID

### 6.1 人物 crop 如何生成

程序遍历 masks/person_tracks.json 中每个稳定 track：

1. 从 frames/ 读取原始 RGB 帧；
2. 从 masks/instances 读取对应人物 mask；
3. 按 bbox 裁剪人物；
4. 将 mask 作为 alpha 通道，生成 RGBA PNG；
5. 按分割置信度、人物可见面积、清晰度和是否贴边计算质量分；
6. 默认每个 track 最多选择 5 张、帧间隔至少 8 帧的代表 crop；
7. 最终选择的 appearance_image 会作为 Qwen 的人物参考图。

相关文件保存在：

~~~text
identity/track_XXXX/crop_fXXXXXX_qN.png
identity/track_observations.json
identity/bindings.json
~~~

### 6.2 PERSON_* ID 如何产生

如果 track 成功绑定到角色 registry：

~~~text
friends:monica  -> PERSON_MONICA
friends:chandler -> PERSON_CHANDLER
~~~

如果未达到身份阈值或没有 registry：

~~~text
track fallback index 1 -> PERSON_1
track fallback index 2 -> PERSON_2
~~~

Qwen 只能使用程序提供的这些 ref_id，不能自行创造人物 ID。

### 6.3 自动身份绑定的接受条件

默认配置：

- 最佳相似度至少 0.90；
- 最佳与第二名的 margin 至少 0.08；
- 同一个 shot 内不允许两个自动 track 绑定到同一 canonical identity；
- 不满足条件时保守回退为 anonymous PERSON_N；
- anonymous 不会阻止 Schema 通过，但最终 document status 会变为 needs_review。

histogram 后端只适合工程链路验证，不应当当作可靠人脸身份识别。正式生产建议使用 InsightFace 或人工绑定。

### 6.4 人工身份覆盖

manual_bindings.json 示例：

~~~json
{
  "bindings": [
    {"track_id": 0, "canonical_identity": "friends:monica"},
    {"track_id": 1, "canonical_identity": "friends:chandler"}
  ]
}
~~~

运行时加入：

~~~bash
--manual-bindings /path/to/manual_bindings.json
~~~

人工绑定适合修复：

- 脸部遮挡导致 InsightFace 无检测；
- 多个角色服装相近；
- track 短、模糊或侧脸；
- 自动结果低于阈值但人工可以明确确认。

## 7. 发送给 Qwen 的请求内容

### 7.1 多模态内容顺序

QwenProvider 按以下顺序组装 user content：

1. 文本标签 [TARGET_VIDEO_WITH_ORIGINAL_AUDIO]；
2. video_url，值为 target MP4 的 Base64 data URI；
3. 每个人物的文本标签 [REFERENCE PERSON_*]；
4. 对应人物 appearance_image 的 image_url Base64 data URI；
5. 包含 timeline、reference_bank 和输入顺序说明的 user_text。

previs video、camera track、actor track、render job 和 observation 帧都不会加入默认 Omni 请求。

### 7.2 user_text 中的固定信息

user_text 会提供：

~~~json
{
  "timeline": {
    "duration_sec": 8.0,
    "fps": 24.0,
    "frame_count": 192,
    "interval_convention": "[start_sec,end_sec)"
  },
  "reference_bank": [
    {
      "ref_id": "PERSON_MONICA",
      "appearance_image": "identity/...png",
      "canonical_identity": "friends:monica",
      "source_track_id": 0
    }
  ],
  "input_order": {
    "video": "TARGET_VIDEO_WITH_ORIGINAL_AUDIO",
    "reference_images": ["PERSON_MONICA"]
  }
}
~~~

其中 duration/FPS/frame_count 是程序提供的权威值，模型不应自行估计总时长。

### 7.3 system prompt 的职责约束

omni_joint_v1 要求模型：

- 只使用已提供的 ref_id；
- 镜头区间有序、不能重叠，并覆盖完整视频；
- 对白保留原语言逐字文本；
- 对白 speaker 无法可靠确定时必须为 null，不能猜测；
- 非对白事件不能设置 speaker；
- 每个 shot 必须列出 active_event_ids；
- 不虚构对白、旁白、音乐和音效；
- 不输出 3D 几何、相机矩阵或骨骼运动；
- 不描述 proxy cylinder、方块、mask、分割伪影和占位外观。

### 7.4 HTTP 请求参数

默认请求主要参数：

~~~json
{
  "model": "qwen3.8-omni-flash",
  "temperature": 0.0,
  "max_tokens": 8192,
  "modalities": ["text"],
  "reasoning_effort": "none",
  "response_format": {"type": "json_object"}
}
~~~

如果配置 structured_output=json_schema，则会把 proposal Schema 作为 strict JSON Schema response_format 发送；配置为 none 时不发送 response_format。

### 7.5 10 MiB Base64 内联限制

当前适配器把每个视频或图片完整读入内存并转换成 data URI。转换后的单个 data URI 必须小于 10 MiB，否则在发请求前直接抛出：

~~~text
Inline media exceeds Qwen's 10 MiB Base64 limit
~~~

由于 Base64 会把体积放大约三分之一，原始 MP4 的安全上限显著低于 10 MiB，通常应控制在约 7 MiB 量级并预留 MIME/data URI 开销。这里是当前客户端实现限制，不代表服务端所有上传方式的通用限制。

如果超限，可选择：

- 重新编码 clip，降低码率或分辨率，但保持正确音轨；
- 进一步切短视频；
- 改造 provider 使用服务端支持的对象存储 URL；
- 改走 observation 多图 + transcript 的分离式路径。

## 8. Qwen proposal 数据结构

模型必须输出一个 JSON object，顶层固定为：

~~~json
{
  "references": [],
  "shots": [],
  "events": [],
  "global": {}
}
~~~

### 8.1 references

每个已提供人物必须出现一次：

~~~json
{
  "ref_id": "PERSON_MONICA",
  "semantic_description": "Monica stands near the apartment doorway.",
  "id_features": "Dark shoulder-length hair and a blue top.",
  "attributes": {
    "clothing": "blue top",
    "hairstyle": "dark shoulder-length hair",
    "accessories": null
  },
  "confidence": 0.95
}
~~~

### 8.2 shots

~~~json
{
  "shot_id": "SHOT_1",
  "start_sec": 0.0,
  "end_sec": 3.2,
  "visual_description": "PERSON_MONICA turns toward PERSON_CHANDLER as he enters.",
  "camera": {
    "shot_type": "medium two-shot",
    "angle": "eye level",
    "movement": "static"
  },
  "references_in_shot": ["PERSON_MONICA", "PERSON_CHANDLER"],
  "active_event_ids": ["EVENT_1"],
  "confidence": 0.9
}
~~~

shots 至少有一条，必须按时间排序、不能互相重叠，并从接近 0 秒覆盖到接近视频结尾。

### 8.3 events

对话事件：

~~~json
{
  "event_id": "EVENT_1",
  "type": "dialogue",
  "start_sec": 0.4,
  "end_sec": 1.8,
  "speaker": "PERSON_MONICA",
  "line": "Hey, where are you going?",
  "description": null,
  "speaker_evidence": {
    "visible_on_screen": true,
    "mouth_activity": true,
    "voice_continuity": null
  },
  "confidence": 0.92
}
~~~

音效事件：

~~~json
{
  "event_id": "EVENT_2",
  "type": "sfx",
  "start_sec": 2.2,
  "end_sec": 2.5,
  "speaker": null,
  "line": null,
  "description": "A door closes.",
  "speaker_evidence": {
    "visible_on_screen": null,
    "mouth_activity": null,
    "voice_continuity": null
  },
  "confidence": 0.8
}
~~~

允许的 event type 为 dialogue、sfx、music。

### 8.4 global

~~~json
{
  "scene_description": "A warmly lit apartment living room.",
  "global_style": "Live-action multi-camera sitcom.",
  "global_audio": {
    "background_music": null,
    "ambient_sound": "Interior room tone."
  }
}
~~~

静音片段应当返回空 events，且 background_music 和 ambient_sound 均为 null，不应为了填字段虚构声音。

## 9. 输出归一化

模型返回值并不直接写入最终数据。normalize_omni_proposal 会把常见别名转换成冻结结构，例如：

- reference_bank -> references；
- annotations.shots -> shots；
- annotations.events -> events；
- sound_effect / sound effect / effect -> sfx；
- speech -> dialogue；
- time_range: [start, end] -> start_sec/end_sec；
- description -> visual_description；
- shot_scale -> camera.shot_type；
- camera_motion -> camera.movement；
- speaker_ref_id -> speaker；
- text / transcript -> line；
- setting / setting_summary -> scene_description；
- visual_style -> global_style。

归一化会提供少量保守默认值，但不会绕过校验。例如 references 缺少某个已提供 PERSON_* 时，后续校验仍然失败并要求模型重试。

需要注意一个行为：如果某个 shot 没有 references_in_shot，归一化会先从描述中搜索 PERSON_*；仍找不到时会暂时填入所有 allowed_ref_ids。因此人工复核时要重点检查镜头人物列表是否过宽。

## 10. 校验与自动重试

### 10.1 JSON 解析

解析器支持：

- 纯 JSON object；
- Markdown fenced JSON；
- JSON 前后有少量说明文字时，截取第一个 { 到最后一个 }。

最终根节点必须是 object，否则失败。

### 10.2 Schema 与语义校验

proposal 至少执行以下检查：

- 所有必需字段存在，类型符合 qwen_omni_mtss_proposal_v1；
- reference ref_id 不重复；
- 不允许未知 ref_id；
- 不允许漏掉程序提供的 ref_id；
- event_id、shot_id 不重复；
- event/shot 时间在 [0, duration] 内；
- dialogue 必须有逐字 line；
- speaker 只能是允许的 ref_id 或 null；
- 非 dialogue 事件 speaker 必须为 null；
- shot 不能与前一 shot 重叠；
- shot 中 active_event_ids 必须真实存在；
- shot 关联的 event 必须与该 shot 时间区间重叠；
- 每个 event 至少被一个 shot 关联；
- 第一条 shot 必须在 0 秒附近开始；
- 最后一条 shot 必须在视频结尾附近结束。

### 10.3 两层重试

当前有两层重试：

1. HTTP/解析层：max_retries=2，最多 3 次网络尝试；
2. 语义校验层：semantic_retries=2，最多生成 3 个 proposal。

因此极端情况下，一个 clip 可能发生最多 9 次 HTTP 尝试。HTTP 408、409、429 和 5xx 会重试；其他 4xx 通常立即停止。网络重试采用短指数退避。

proposal 未通过语义校验时，程序会把具体错误追加到下一次 user prompt，要求模型重新输出完整 JSON，而不是只返回补丁。

### 10.4 raw 调试文件

每次尝试都会保存：

~~~text
raw/omni_joint_request.json
raw/omni_joint_response_attempt_1.json
raw/omni_joint_normalized_attempt_1.json
raw/omni_joint_metadata_attempt_1.json
raw/omni_joint_response_attempt_2.json
...
~~~

metadata 中包含：

- model；
- 第几次 HTTP attempt；
- latency_sec；
- token usage（如果后端返回）；
- response_id；
- video_count 和 image_count；
- semantic_validation_errors。

这些文件是定位模型输出错误、性能长尾和重试率的第一现场。

## 11. 与 StageRecon 几何信息合并

Qwen proposal 通过后，merge_mtss3d 会生成最终 mtss_3d_v1：

- Qwen references -> mtss.references；
- Qwen shots -> mtss.shots；
- Qwen events -> mtss.events；
- Qwen global -> mtss.global；
- SAM3 track 和角色绑定 -> extensions.mtss_3d.entity_bindings；
- 演员根节点与轨迹约束 -> extensions.mtss_3d.blocking_motion；
- camera/actor/render 文件及哈希 -> extensions.mtss_3d.geometry；
- target video、人物 crop 和 geometry 来源 -> extensions.mtss_3d.evidence。

Qwen 给出的 shot/event 秒数会转换为帧范围，并标记 temporal_precision=sampled_estimate。它不是几何真值，也不应当被解释为帧级精确边界。

### 11.1 最终 status

正式运行的 status 规则：

- complete：没有匿名人物、没有缺失 reference、所有 dialogue 都有已知 speaker；
- needs_review：存在 anonymous identity、Qwen 漏 reference 后插入保守占位，或 dialogue speaker 为 null；
- dry_run_placeholder：使用 --dry-run。

validation_report.json 的 passed/failed 与 document status 是两个概念：

- passed 表示结构和交叉引用合法；
- needs_review 表示结构可以合法，但语义上仍需要人工确认；
- 只有 status=complete 才能通过 export 的 --require-complete 门槛。

## 12. 输出目录说明

一次完整运行的输出类似：

~~~text
captions/mtss_3d_v1/
  input_manifest.json
  mtss_3d.json
  validation_report.json
  training_record.json
  run_manifest.json
  review.html
  observations/
    observations.json
    frames/...
  identity/
    bindings.json
    track_observations.json
    track_0000/...
  geometry/
    ...冻结的 camera/actor/render 文件...
  semantic/
    omni_joint_proposal.json
  raw/
    omni_joint_request.json
    omni_joint_response_attempt_*.json
    omni_joint_normalized_attempt_*.json
    omni_joint_metadata_attempt_*.json
~~~

关键文件用途：

| 文件 | 用途 |
|---|---|
| input_manifest.json | 输入路径、哈希、target 时间轴和 geometry 时间轴 |
| identity/bindings.json | track 到 PERSON_* / canonical identity 的绑定与候选分数 |
| semantic/omni_joint_proposal.json | 通过校验后的 Qwen 标准化 proposal |
| mtss_3d.json | 最终四流语义 + StageRecon 扩展 |
| validation_report.json | Schema、ID、时间、文件存在性和 SHA-256 检查 |
| training_record.json | 下游训练直接使用的 previs、reference bank、prompt 和 target |
| review.html | 人工浏览人物、镜头、事件和校验结果 |
| run_manifest.json | 本次配置、模型、request hash、状态和主要输出索引 |

request hash 会覆盖 system prompt、user text、model、所有人物图和视频内容哈希，用于识别请求输入是否真正变化。

## 13. 批量处理

### 13.1 JSONL manifest

每行一个 JSON object。默认 Omni 路径推荐：

~~~json
{"sample_id":"clip_0001","clip_run":"/abs/path/clip_run_0001","previs_video":"/abs/path/clip_run_0001/camera_track_playblast.mp4","target_video":"/abs/path/clips/clip_0001.mp4"}
{"sample_id":"clip_0002","clip_run":"/abs/path/clip_run_0002","previs_video":"/abs/path/clip_run_0002/camera_track_playblast.mp4","target_video":"/abs/path/clips/clip_0002.mp4","manual_bindings":"/abs/path/manual/clip_0002.json"}
~~~

每条记录可使用的字段包括：

- sample_id；
- sequence_id；
- clip_run；
- previs_video；
- target_video；
- output_dir；
- registry；
- audio；
- audio_from_source；
- transcript；
- manual_bindings。

注意：batch 入口当前不从单条 manifest 读取 omni_proposal、visual_proposal 或 audio_proposal。如果批处理需要这些预计算结果，应扩展 tools/run_mtss3d_batch.py 或逐条运行单片入口。

### 13.2 批量命令

~~~bash
cd /root/userdata/300workspace/StageRecon
export DASHSCOPE_API_KEY=YOUR_KEY

python tools/run_mtss3d_batch.py \
  --manifest manifests/caption_jobs.jsonl \
  --output-root outputs/captions/mtss_3d_v1 \
  --registry configs/characters/friends.local.json \
  --identity-backend insightface \
  --model qwen3.8-omni-flash \
  --max-workers 2 \
  --no-resume
~~~

batch 使用 ThreadPoolExecutor。每个 sample 独立捕获异常，单条失败不会中止其他样本。最终生成：

~~~text
manifests/caption_jobs.jsonl.caption_report.json
~~~

### 13.3 并发建议

每个 worker 会把完整 target video 和人物参考图读入内存并 Base64 编码，同时可能执行最多三轮语义请求。因此建议：

1. 首轮 20 条使用 max-workers=1；
2. 确认 API 限流、平均延迟、超时率和内存后再提高到 2–4；
3. 遇到 429 时降低并发，不要单纯增加重试；
4. 对大批量任务按剧集/场景拆 manifest，方便重跑和审计。

## 14. 校验、人工复核与训练集导出

### 14.1 单独校验

~~~bash
python tools/validate_mtss3d.py \
  --input outputs/captions/mtss_3d_v1/clip_0001/mtss_3d.json
~~~

校验不仅看 JSON Schema，还会检查：

- asset 文件是否存在；
- asset SHA-256 是否匹配；
- geometry 文件是否存在且哈希匹配；
- Reference、Shot、Event 的交叉引用；
- 时间和帧范围；
- entity binding 与 Reference 集合是否一致；
- 同一 shot 是否重复绑定同一 canonical identity；
- 没有 audio asset 时是否错误保留了音频事件；
- shot 描述是否出现 proxy cylinder/box 等不应进入语义的数据。

### 14.2 生成人工复核页面

~~~bash
python tools/build_caption_review.py \
  --input outputs/captions/mtss_3d_v1/clip_0001/mtss_3d.json
~~~

建议人工至少核对：

- PERSON_* 与人物参考图是否一致；
- anonymous 是否可以人工绑定角色；
- shot cut、shot scale 和 camera movement 是否正确；
- 动作是否按时间顺序，是否有幻觉；
- 对白是否逐字、语言是否保持原样；
- speaker 是否与嘴部活动一致；
- 短促音效和短对白是否漏标；
- active_events 是否挂在正确 shot；
- 环境声和音乐是否真实存在。

### 14.3 导出训练 JSONL

~~~bash
python tools/export_mtss3d_training_jsonl.py \
  --input-root outputs/captions/mtss_3d_v1 \
  --output datasets/mtss_3d_v1/train.jsonl \
  --require-complete
~~~

--require-complete 会拒绝 needs_review 和 dry_run_placeholder，同时为被拒绝样本生成详细 report。

## 15. 质量门槛建议

### 15.1 20 条工程验证集

先验证：

- 输入路径和 ffprobe 可用；
- target/previs 没有混用；
- API Key、模型名和 base URL 正确；
- 人物 crop 清晰且 alpha mask 正常；
- raw response、normalized proposal、MTSS 和 review 全部落盘；
- validation_report 无错误。

### 15.2 100 条语义评估集

建议统计：

- validation 首次通过率；
- 平均 semantic retry 次数；
- HTTP 超时率和 429 比例；
- 总延迟 P50/P90/P99；
- anonymous identity 比例；
- dialogue speaker=null 比例；
- 人物混淆率；
- 对白字错率和漏句率；
- shot boundary 平均偏差；
- 短事件漏检率；
- 动作幻觉率；
- needs_review 比例和人工修订耗时。

### 15.3 扩展到正式批量

在扩大到 500–1,000 条之前应冻结：

- qwen model ID 和 base URL；
- omni_joint_v1 prompt；
- qwen_omni_mtss_proposal_v1 Schema；
- mtss_3d_v1 Schema；
- character registry 版本；
- 身份后端与阈值；
- target 视频编码规范；
- 人工复核标准。

## 16. 常见故障排查

### 16.1 Missing environment variable / Qwen API key is empty

原因：DASHSCOPE_API_KEY 未设置，或自定义 config 中 api_key_env 名称不同。

检查：

~~~bash
test -n "$DASHSCOPE_API_KEY" && echo key-present
~~~

### 16.2 Target video has no video stream / duration is unavailable

原因：target 路径错误、文件损坏，或 ffprobe 无法解析。

检查：

~~~bash
ffprobe -v error -show_streams -show_format /path/to/target.mp4
~~~

### 16.3 Inline media exceeds Qwen's 10 MiB Base64 limit

原因：视频或某个人物参考图转换为 data URI 后超过客户端限制。

处理：压缩/缩短 target，压缩异常大的 PNG crop，或改造为 URL 上传。

### 16.4 Qwen HTTP 401/403

原因：Key 无效、权限不足、base URL 与 Key 不匹配。

处理：确认 Key 所属服务和地区端点。非 408/409/429 的 4xx 通常不会自动重试。

### 16.5 Qwen HTTP 429

原因：请求频率、并发或配额受限。

处理：降低 --max-workers，分批运行，并检查服务配额。

### 16.6 request timed out

默认 timeout 为 600 秒。如果频繁超时，应先检查视频编码、大小和服务状态，而不是无限加大 timeout。可用 tools/probe_qwen38_omni_video.py 对代表性输入单独探测延迟和音轨能力。

### 16.7 proposal remained invalid after 3 attempts

先查看：

~~~text
raw/omni_joint_metadata_attempt_*.json
raw/omni_joint_normalized_attempt_*.json
~~~

常见原因：

- 漏掉某个 PERSON_* reference；
- 创造了未知 speaker；
- dialogue 没有 line；
- shot 重叠或没有覆盖视频结尾；
- event 没有关联到任何 shot；
- active_event_ids 指向不存在或时间不重叠的 event。

### 16.8 status 一直是 needs_review

检查 mtss_3d.json：

- extensions.mtss_3d.entity_bindings 是否含 anonymous；
- mtss.events 中 dialogue speaker 是否为 null；
- annotation_meta.notes 是否记录 Qwen 漏 reference。

解决方式通常是补人工身份绑定、修订 proposal，或改进 reference crop 后使用 --no-resume 重跑。

### 16.9 修改了配置但结果没变化

通常是 resume 命中了已有 mtss_3d.json。使用 --no-resume，必要时换一个新的 --output-dir 保留旧版本用于对比。

### 16.10 外部音频没有被 Qwen 听到

这是当前实现的预期行为：Omni 请求只发送 target MP4。请把音频 remux 进 target，或提供 timestamped transcript 进入分离式路径。

## 17. Qwen3.8-Omni 输入能力探测

仓库提供独立探针：

~~~bash
python tools/probe_qwen38_omni_video.py \
  --manifest manifests/qwen38_probe.json \
  --output outputs/qwen38_probe/report.json
~~~

manifest 示例：

~~~json
{
  "cases": [
    {
      "case_id": "real_clip_with_audio",
      "video": "/abs/path/to/clip.mp4",
      "prompt": "Describe the visual sequence and transcribe any audible dialogue.",
      "expected": "Manual expectation for later comparison"
    }
  ]
}
~~~

探针使用流式 chat/completions，记录视频字节数、延迟、usage、响应模型、文本和 reasoning content，适合在正式批量前确认当前 endpoint 的真实能力。

docs/qwen_video_input_probe_2026-10-22.md 记录的是另一个通用 qwen 接口的历史实测，不是当前 qwen3.8-omni-flash 北京端点的等价测试。历史文档关于“音轨不可用”的结论不能直接外推到当前模型，但它关于稀疏抽帧、短事件漏检、时间定位粗和延迟长尾的风险仍值得作为质检假设重新验证。

## 18. 数据安全与可追溯性

默认 Omni 路径会把以下数据发送到配置的远程模型服务：

- 完整 target video；
- target MP4 中的音轨和对白；
- 从真实帧和 mask 生成的人物参考 crop；
- canonical identity（如果 registry 已成功绑定）；
- 视频时长、FPS、帧数和 track ID。

因此生产使用前应确认数据授权、跨区域传输要求、供应商数据保留策略和访问审计要求。

本地输出 raw/、identity/ 和 review.html 可能包含可识别人物、完整对白和模型原始响应，不应上传到公开仓库。API Key 应只通过环境变量或安全凭据系统提供，不要写入 config、manifest、run_manifest 或脚本。

StageRecon 为输入 asset、geometry 文件和模型 request 计算 SHA-256，可用于追踪“同一个 sample 是否使用了相同媒体、prompt、模型和人物参考图”，但哈希不能替代数据访问控制。

## 19. 推荐标准操作流程

1. 运行 clip reconstruction，确认 frames、person_tracks、camera_track 和 actor_tracks 完整。
2. 准备角色 gallery 并生成本地 registry。
3. 对单条 clip 做 dry-run，检查人物 crop、bindings 和 geometry。
4. 用 qwen38 probe 对代表性视频确认当前 endpoint 的视频、音轨和延迟表现。
5. 正式运行单条 Omni caption，并检查 raw、semantic、validation 和 review。
6. 对匿名身份和未知 speaker 做人工覆盖或修订。
7. 用 20 条验证工程稳定性，再用 100 条统计语义质量与重试率。
8. 冻结模型、prompt、Schema、registry 和视频编码规范。
9. 小并发批量运行，持续监控 429、超时、invalid proposal 和 needs_review。
10. 只有 validation passed、人工复核完成且 status=complete 的数据，才用 --require-complete 导出到正式训练集。

## 20. 开发自检

修改 caption pipeline、prompt、Schema 或 provider 后，至少运行：

~~~bash
cd /root/userdata/300workspace/StageRecon

python -m compileall -q stagerecon tools tests
python -m unittest discover -s tests -v
~~~

如果修改了 Omni 字段、别名或校验规则，应同步更新：

- schemas/qwen_omni_mtss_proposal_v1.schema.json；
- stagerecon/captioning/omni_normalize.py；
- stagerecon/captioning/validation.py；
- stagerecon/captioning/merge.py；
- prompts/mtss_3d/omni_joint_v1.txt；
- tests/test_mtss3d_captioning.py；
- 本文档。
