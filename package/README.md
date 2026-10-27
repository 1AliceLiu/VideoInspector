# VideoInspector API 交付包 v0.2

这是给同事的“私有化 Docker 包”：包含现有检测程序、控制面 API、配置模板和说明。
同事拿回去后，用自己的 GPU 服务器运行；检测算法和模型仍然在本地，不经过你们的服务器。

> 如果你其实想要“托管 API”（程序跑在你们服务器，同事远程调用），同一套代码也可以直接部署在你们服务器上，只是网络入口和运维方式不同。这个包先按私有化交付做。

## 1. 包里有什么

```text
package/
├── docker-compose.yml        # 编排：detector + api 两个容器
├── .env.example              # 环境变量模板（复制成 .env）
├── config/
│   ├── config.example.yaml   # 配置模板（复制成 config.yaml）
│   └── config.yaml           # 实际配置（init.sh 自动生成）
├── models/                   # 模型文件（从主项目 deploy/models/ 复制）
├── generated/                # TensorRT 引擎等构建产物（自动生成）
├── logs/                     # 检测日志
├── runtime/                  # 状态/任务/配置历史/重载标记
├── docs/API-v1.md            # 给同事看的接口文档
└── scripts/
    ├── init.sh               # 初始化目录和模板
    ├── build.sh              # 构建镜像
    ├── start.sh              # 启动
    ├── stop.sh               # 停止
    └── logs.sh               # 看日志
```

## 2. 两个容器分别是什么

| 模块 | 位置 | 大白话解释 |
| --- | --- | --- |
| `detector` 容器 | `deploy/Dockerfile` + `server/` | 真正干活的检测程序：拉 RTSP、跑模型、判断违规、推 Webhook、写状态 |
| `api` 容器 | `deploy/Dockerfile.api` + `server/api/` | 控制台：同事通过它加摄像头、绑算法、查状态、测试 RTSP；不跑模型，不需要 GPU |

为什么要拆两个容器：

- 检测容器依赖 DeepStream/GPU，升级风险高；API 容器只读写配置，升级简单。
- API 挂了不影响检测；检测重启时 API 还能继续接请求（变更会排队/等就绪）。
- 同事只需要访问 API 端口，不需要碰 GPU 容器。

## 3. 代码模块说明（`server/api/`）

| 文件 | 作用 | 你可以把它理解成 |
| --- | --- | --- |
| `control_plane.py` | HTTP 路由、鉴权、请求/响应 | 前台接待，决定“谁可以办什么事” |
| `config_store.py` | 读写 `config.yaml`、校验、备份、生成按摄像头规则 | 配置管家，负责把同事的要求翻译成检测程序能读的配置 |
| `operations.py` | 异步任务队列、等待管线就绪、失败回滚 | 施工队，按顺序改配置、重启、发现失败就恢复上一版 |
| `probe.py` | RTSP 探测（优先 ffprobe，缺了用 RTSP OPTIONS） | 体检医生，先看看摄像头能不能拉到画面 |
| `status_writer.py` | 检测进程把在线状态、帧率、最后告警写进 `runtime/status.json` | 仪表盘，API 从这里读状态 |
| `util.py` | 时间、RTSP 密码脱敏、原子写文件 | 小工具 |
| `errors.py` | 统一错误码 | 统一说法，方便同事排障 |

## 4. 检测进程改了哪里

没有改检测算法，只加了配置/状态/告警输出相关的接口：

| 位置 | 改动 | 作用 |
| --- | --- | --- |
| `server/main.py` | 监督循环监听 `runtime/reload.request` | 收到 API 的重载通知后，优雅重启检测子进程 |
| `server/service.py` | 启动后写 `runtime/ready` 和 `runtime/status.json` | 让 API 知道“新配置已经起来了” |
| `server/pipeline/health.py` | 增加 `stats()` 只读统计 | 给状态接口提供逐路帧数和帧间隔 |
| `server/alert/rules.py` | 增加可选 `alert_type` | 按摄像头生成的规则对外仍报基础算法名 |
| `server/pipeline/probe.py` | OSD 标签改用 `alert_type` | 预览画面不泄漏 `__cam__` 内部规则名 |
| `server/pipeline/builder.py` | 透传设备编码/通道号、Webhook 密钥、默认规则过滤 | 告警带 ICC 信息，未配置 active_rules 的摄像头不串参数 |
| `server/alert/manager.py` + `webhook.py` | 告警加 `event_id`、设备编码/通道号、签名头 | 方便同事平台去重、对账、防伪造 |

检测判定逻辑、模型拓扑、DeepStream 管线都没有改。

## 5. 数据怎么流

```text
同事平台
   │ ① POST /v1/cameras（RTSP + 算法）
   ▼
api 容器 ──写──> config/config.yaml
   │ ② 写 runtime/reload.request
   ▼
detector 容器（main.py 监督循环）
   │ ③ 重启检测子进程，加载新配置
   │ ④ 拉 RTSP、推理、判定
   ├── ⑤ 写 runtime/status.json（在线/帧率/最后告警）
   └── ⑥ POST Webhook 到同事平台（告警 + 证据帧）
```

## 6. 第一次启动

### 6.0 前置条件

- NVIDIA GPU（显存/算力按实际路数评估）
- NVIDIA 驱动 + NVIDIA Container Toolkit
- Docker + Docker Compose v2
- 能拉取 `nvcr.io/nvidia/deepstream:9.0-triton-multiarch`（需要 NGC 账号；离线环境先 `docker load`）
- 模型文件：放到 `package/models/`

### 6.1 准备模型

把主项目的模型目录复制过来：

```bash
cp -a ../deploy/models/. models/
```

目录结构保持：

```text
models/person/...
models/helmet/...
models/vest/...
models/harness/...
models/smoke/...
```

没有模型时，检测容器首次启动会失败；这是正常的。

### 6.2 初始化配置

```bash
cd package
cp .env.example .env
cp config/config.example.yaml config/config.yaml
# 或者：./scripts/init.sh
```

修改：

- `.env` 里的 `SAFETY_API_KEY`（必须改成随机长字符串）
- `config/config.yaml` 里的 RTSP 地址、Webhook 地址
- 如果走 ICC：设置 `SAFETY_ICC_RTSP_TEMPLATE`，例如
  `rtsp://icc.example.com:554/{device_code}/{channel_no}`

### 6.3 构建并启动

```bash
chmod +x scripts/*.sh
./scripts/build.sh
./scripts/start.sh
```

首次启动会编译 TensorRT 引擎，可能需要几分钟到十几分钟；看日志：

```bash
./scripts/logs.sh
```

API 默认地址：`http://<服务器IP>:18080`

### 6.4 确认 API 活着

```bash
curl http://127.0.0.1:18080/healthz
```

返回类似：

```json
{"status":"ok","pipeline":"running","config_revision":1}
```

## 7. API 怎么用

先设置两个变量：

```bash
export API=http://127.0.0.1:18080
export KEY=change-me-to-a-long-random-string
```

### 7.1 看有哪些算法

```bash
curl -H "Authorization: Bearer $KEY" $API/v1/algorithms
```

### 7.2 加摄像头

```bash
curl -X POST $API/v1/cameras \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "1824",
    "name": "ZoneC_Outdoor_001",
    "rtsp_url": "rtsp://user:pass@camera.example.com:554/stream1",
    "algorithms": [
      {"id": "no_helmet", "params": {"cooldown_seconds": 10, "min_detection_count": 3, "attribute_threshold": 0.5}},
      {"id": "no_vest", "params": {"cooldown_seconds": 600, "min_detection_count": 20}}
    ]
  }'
```

返回 `202` 和一个 `operation_id`。因为要重建管线，变更不会立刻完成。

### 7.3 查变更结果

```bash
curl -H "Authorization: Bearer $KEY" $API/v1/operations/<operation_id>
```

`status` 变成 `succeeded` 就生效了；`rolled_back` 表示失败并恢复了上一版。

### 7.4 先测试再启用

```bash
# 1. 先建一个停用的摄像头
curl -X POST $API/v1/cameras ... -d '{"enabled": false, ...}'

# 2. 探测 RTSP
curl -X POST $API/v1/cameras/1824/probe -H "Authorization: Bearer $KEY"

# 3. 看探测结果；确认可达后再启用
curl -X PATCH $API/v1/cameras/1824 \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

### 7.5 给已有摄像头换算法

```bash
curl -X PUT $API/v1/cameras/1824/algorithms \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"algorithms": [{"id": "no_helmet"}, {"id": "smoking"}]}'
```

### 7.6 看状态

```bash
curl -H "Authorization: Bearer $KEY" $API/v1/cameras
```

每路会带：

- `online`
- `last_frame_age_seconds`
- `fps`
- `last_alert_at`
- `algorithm_versions`

### 7.7 画区域（区域入侵 / 限定算法区域）

区域坐标用归一化 0~1（相对画面宽高），可直接填平台标定文件里的 `Points[].X/Y`。下面把摄像头 2628 的区域 `r9-1` 绑定为区域入侵，同时保留全图安全帽检测（摄像头需已存在）：

```bash
curl -X PUT $API/v1/cameras/2628/algorithms \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"algorithms": [
        {"id": "no_helmet"},
        {"id": "intrusion", "params": {"zones": [
          {"id": "r9-1", "points": [[0.2943,0.3311],[0.4398,0.3408],[0.5252,0.6883],[0.5066,0.9835],[0.0055,0.9874],[0.0044,0.6981]]}
        ]}}
      ]}'
```

返回 `202` 后用 `operation_id` 查结果。区域的完整规则（锚点、去抖、错误码）见 `docs/API-v1.md` 第 9.2 节。

## 8. v0.2 做了什么、没做什么

### 已实现

- **区域限制（zones）**：任意算法可绑定 0..N 个归一化多边形；不绑定 = 全图判定
- **区域入侵（intrusion）**：区域内有人即告警；同一次进入只报 1 条，离开后再进入报新的 1 条（`vacant_frames` 去抖）
- 告警带 `zones`（回指标定的 `RuleId`）和 `objects[].attributes[].zone`
- 摄像头增删改查、启停
- 绑定/替换已有算法
- 按摄像头调 `cooldown_seconds`、`min_detection_count`
- detector 类算法按摄像头调 `attribute_threshold`（安全帽、抽烟）
- RTSP 探测
- 异步变更 + 等待就绪 + 失败自动回滚
- API Key 鉴权
- 状态查询（在线、帧间隔、帧率、最后告警、算法版本）
- ICC 设备编码/通道号 + 转发 RTSP 两种来源
- Webhook `event_id`、`schema_version`、设备编码/通道号、HMAC 签名

### 8.1 从 v0.1 升级到 v0.2

1. 替换交付包；模型包不变，继续使用 v0.1 的模型包。
2. v0.1 的 `config/config.yaml` 没有 `intrusion` 模板规则。要使用区域入侵，把 `config/config.example.yaml` 里的 `intrusion` 段复制到 `config/config.yaml` 的 `rules:` 下面。这一步不会改变现有摄像头的判定，因为没有摄像头引用它。
3. 执行 `./scripts/build.sh && ./scripts/start.sh`（镜像标签变为 `0.2`）。
4. `curl -H "Authorization: Bearer $KEY" $API/v1/algorithms` 能看到 `intrusion`，说明模板已生效。

### 暂未实现

- **classifier 类算法（反光衣/安全带）按摄像头选 `profile`（松/正常/严）**
  - 当前只支持按摄像头调冷却和连续帧数；
  - `profile` 会返回 `422 UNSUPPORTED_PARAM`；
  - 原因是分类器阈值写在模型配置里，按摄像头改需要增加模型实例或改推理链，放到下一版。
- 新算法/新模型上传、编译、上架
- 多租户自助开通（本包按单租户设计）
- 单帧图片推理 API
- Webhook 持久化队列（当前是进程内重试，至少一次投递）

## 9. 常见问题

### API 返回 401

`.env` 里的 `SAFETY_API_KEY` 和请求头不一致。改完 `.env` 后执行 `./scripts/start.sh` 重建容器。

### operation 一直是 pending / applying

看检测容器日志：

```bash
./scripts/logs.sh
```

常见原因：模型文件缺失、TensorRT 引擎编译失败、RTSP 拉不到、GPU 不可用。

### operation 变成 rolled_back

新配置没能在超时时间内生效，系统已自动恢复上一版。看 `error.message` 和检测日志。

### probe 失败

- 确认服务器能访问摄像头/转发服务的 RTSP 地址；
- ICC 模式确认 `SAFETY_ICC_RTSP_TEMPLATE` 正确；
- 先用 `ffprobe` 在容器里手动试：
  ```bash
  docker exec -it video-inspector-api ffprobe -v error -rtsp_transport tcp <rtsp_url>
  ```

### 第一次启动很慢

正常。`generated/` 为空时要编译 TensorRT 引擎；之后重启会跳过。

## 10. 安全注意事项

1. **必须改 `SAFETY_API_KEY`**，不要用模板值。
2. API 建议放在反向代理后面，只开 HTTPS，并限制来源 IP。
3. `config/config.yaml` 里有 RTSP 密码，权限设为 `600`，不要提交到 git。
4. Webhook 建议配置 `SAFETY_WEBHOOK_SECRET`，同事平台校验 `X-Signature`。
5. 不要把 `.env`、`logs/`、真实的 `config.yaml` 打进交付包。

## 11. 怎么发给同事

直接生成交付压缩包：

```bash
cd package
./scripts/make_bundle.sh
```

产物在 `package/dist/`：

```text
video-inspector-api-v0.2.tar.gz
video-inspector-api-v0.2.tar.gz.sha256
```

默认不包含模型（模型体积大、属于算法资产）。如果需要连模型一起打包：

```bash
WITH_MODELS=1 ./scripts/make_bundle.sh
```

同事解压后：

```bash
tar -xzf video-inspector-api-v0.2.tar.gz
cd video-inspector-api-v0.2/package
./scripts/init.sh
```

如果压缩包里没有模型，先把 `models/` 放好，再执行 `./scripts/build.sh && ./scripts/start.sh`。

不要包含：

- `.env`
- `config/config.yaml`（含真实密码）
- `logs/`
- `runtime/`
- `generated/`（可在目标机器重建）

如果同事机器不能联网，先在能联网的机器上构建镜像，再导出：

```bash
docker save video-inspector-detector:0.2 video-inspector-api:0.2 -o video-inspector-v0.2.tar
```

注意：TensorRT 引擎和 GPU/驱动/TensorRT 版本绑定，**不能把 `generated/` 直接拷到不同 GPU 的机器上**，目标机器首次启动会自动重建。
