# 视频检测开放 API v1（待确认）

> 版本：v1.1-draft  
> 日期：2026-10-23  
> 状态：根据同事反馈修订，待最终确认  
> 说明：本文档只定义对外契约，不包含内部实现细节。  
> 第一版 Docker 包（v0.1）已实现大部分接口；classifier 按摄像头 `profile` 暂未开放，详见 `package/README.md`。

## 0. 接口总览

| 方法 | 路径 | 用途 | 返回 |
| --- | --- | --- | --- |
| GET | `/v1/algorithms` | 算法目录 | 200 |
| GET | `/v1/cameras` | 摄像头列表 | 200 |
| GET | `/v1/cameras/{id}` | 摄像头详情 | 200 |
| POST | `/v1/cameras` | 新增摄像头 | 202 + operation_id |
| PATCH | `/v1/cameras/{id}` | 修改摄像头 | 202 + operation_id |
| DELETE | `/v1/cameras/{id}` | 删除摄像头 | 202 + operation_id |
| PUT | `/v1/cameras/{id}/algorithms` | 全量替换绑定算法及参数 | 202 + operation_id |
| GET | `/v1/operations/{id}` | 查询变更结果 | 200 |
| GET | `/healthz` | 健康检查 | 200 |
| POST | `/v1/cameras/{id}/probe` | RTSP 探测 | 202 + operation_id |

## 1. 通用约定

- Base URL：`https://<host>/v1`
- 鉴权：`Authorization: Bearer <API_KEY>`（`/healthz` 除外）
- 编码：`Content-Type: application/json`
- **写操作异步生效**：新增/修改/删除/绑算法返回 `202 Accepted` + `operation_id`，用 `GET /v1/operations/{id}` 查结果。原因是变更要重建检测管线，通常 10~30 秒。
- 参数、权限、容量错误立即返回 `4xx`；管线启动失败在 operation 里返回 `failed` 或 `rolled_back`。
- 同一时刻只允许一个变更生效；并发写返回 `409 OPERATION_IN_PROGRESS`。建议串行调用。
- 写操作支持 `Idempotency-Key` 请求头，相同 key 返回同一个 operation。
- 摄像头 ID 由调用方定义，实例内唯一，`[A-Za-z0-9_-]{1,64}`；会原样出现在 webhook 的 `camera_id`。
- 摄像头可额外携带 ICC 设备编码和通道号；`id` 仍是唯一主键，`device_code`/`channel_no` 用于 ICC 寻址和排障，不要求与 `id` 相同。
- 算法 ID 来自 `GET /v1/algorithms`。
- 本服务按**单租户**部署；如果同事平台要分发给多个下游，由同事平台自行做租户隔离和路由。
- 实例最多 32 路摄像头（引擎硬上限），实际配额另定。

### 1.1 写操作通用响应

```json
{
  "operation_id": "op_20261023_0001",
  "status": "pending"
}
```

### 1.2 错误格式

```json
{
  "error": {
    "code": "CAMERA_EXISTS",
    "message": "camera 1824 already exists",
    "request_id": "req_..."
  }
}
```

常见错误码：

| HTTP | code | 含义 |
| --- | --- | --- |
| 400 | `INVALID_ARGUMENT` | 参数缺失或格式错误 |
| 401 | `UNAUTHORIZED` | API Key 无效 |
| 404 | `NOT_FOUND` | 摄像头或 operation 不存在 |
| 409 | `CAMERA_EXISTS` | 摄像头 ID 重复 |
| 409 | `CAPACITY_EXCEEDED` | 超出摄像头配额 |
| 409 | `OPERATION_IN_PROGRESS` | 已有变更正在生效 |
| 409 | `LAST_CAMERA` | 不能删除最后一个启用中的摄像头 |
| 422 | `ALGORITHM_NOT_FOUND` | 算法 ID 不存在 |
| 422 | `INVALID_PARAM` | 算法参数不合法或该算法不支持该参数 |
| 422 | `UNSUPPORTED_PARAM` | 参数格式合法，但当前版本未开放（如 classifier profile） |
| 429 | `RATE_LIMITED` | 请求过于频繁 |
| 503 | `PIPELINE_NOT_READY` | 检测管线未就绪 |

### 1.3 变更流程

```text
写接口 → 202 + operation_id
      → GET /v1/operations/{id} 轮询
      → pending → applying → succeeded
                          └→ failed / rolled_back
```

失败时自动回滚到上一版配置并重启管线，`error` 字段返回原因。

## 2. 算法目录

### `GET /v1/algorithms`

用途：列出当前可绑定的算法、支持按摄像头覆盖的参数及默认值，供平台渲染配置表单。

```json
{
  "items": [
    {
      "id": "no_helmet",
      "display_name": "安全帽检测",
      "kind": "detector",
      "model_version": "2026.08.04",
      "params_schema": {
        "cooldown_seconds": {"type": "number", "min": 0, "default": 10},
        "min_detection_count": {"type": "integer", "min": 1, "default": 3},
        "attribute_threshold": {"type": "number", "min": 0, "max": 1, "default": 0.5}
      }
    },
    {
      "id": "no_vest",
      "display_name": "未穿反光衣",
      "kind": "classifier",
      "model_version": "2026.09.04",
      "params_schema": {
        "cooldown_seconds": {"type": "number", "min": 0, "default": 300},
        "min_detection_count": {"type": "integer", "min": 1, "default": 15},
        "profile": {"type": "string", "enum": ["loose", "normal", "strict"], "default": "normal", "supported": false}
      }
    }
  ]
}
```

参数说明：

| 参数 | 适用算法 | 说明 |
| --- | --- | --- |
| `cooldown_seconds` | 全部 | 同一摄像头同一规则的告警冷却秒数 |
| `min_detection_count` | 全部 | 连续多少帧命中才触发，越大越稳、越迟钝 |
| `attribute_threshold` | detector 类（如 `no_helmet`、`smoking`） | 违规置信度门限，0~1 |
| `profile` | classifier 类（如 `no_vest`、`no_harness`） | 严格程度档位：`loose` / `normal` / `strict`；**v0.1 暂未开放按摄像头配置** |

**classifier 阈值说明**：分类器的阈值目前写死在模型配置里，v0.1 不支持按摄像头传任意数值，也不支持按摄像头选 `profile`；当前只支持 `cooldown_seconds` 和 `min_detection_count` 按摄像头调。`profile` 字段先保留在契约里，下一版再实现。实现时需要增加全局分类器实例，档位越多 GPU 开销越高，建议最多 3 档。

## 3. 摄像头对象

```json
{
  "id": "1824",
  "name": "ZoneC_Outdoor_001",
  "device_code": "ICC-DEVICE-001",
  "channel_no": "1",
  "source_mode": "icc",
  "rtsp_url": "rtsp://user:***@host/stream",
  "enabled": true,
  "algorithms": [
    {
      "id": "no_helmet",
      "params": {
        "cooldown_seconds": 10,
        "min_detection_count": 3,
        "attribute_threshold": 0.5
      }
    },
    {
      "id": "no_vest",
      "params": {
        "cooldown_seconds": 600,
        "min_detection_count": 20
      }
    }
  ],
  "status": {
    "online": true,
    "last_frame_age_seconds": 2,
    "fps": 24.8,
    "last_alert_at": "2026-10-23T02:15:33Z",
    "algorithm_versions": {
      "no_helmet": "2026.08.04",
      "no_vest": "2026.09.04"
    }
  }
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 摄像头唯一 ID，由调用方定义 |
| `name` | string | 显示名称 |
| `device_code` | string? | ICC 设备编码，可选 |
| `channel_no` | string? | ICC 通道号，字符串以保留前导零，可选 |
| `source_mode` | string | `direct` / `icc` / `forwarded`，默认 `direct` |
| `rtsp_url` | string? | RTSP 地址；响应中密码脱敏。`direct`/`forwarded` 必填；`icc` 可省略，由后端拼接 |
| `enabled` | boolean | 是否接入管线 |
| `algorithms` | object[] | 绑定的算法及按摄像头参数 |
| `status.online` | boolean | 最近 30 秒内是否有帧 |
| `status.last_frame_age_seconds` | number? | 距最近一帧的秒数；从未出帧为 `null` |
| `status.fps` | number | 最近 10 秒滚动平均帧率；停用或离线为 `0` |
| `status.last_alert_at` | string? | 最近一次告警时间（UTC ISO8601）；无告警为 `null` |
| `status.algorithm_versions` | object | 算法 ID → 当前模型版本，用于排障 |

`last_alert_at` 持久化到状态文件，服务重启后不丢（只在告警触发时写入，不影响检测热路径）。

## 4. 摄像头列表

### `GET /v1/cameras`

Query：`enabled`、`algorithm`、`page`、`page_size`

```json
{
  "items": [
    {
      "id": "1824",
      "name": "ZoneC_Outdoor_001",
      "device_code": "ICC-DEVICE-001",
      "channel_no": "1",
      "source_mode": "icc",
      "rtsp_url": "rtsp://user:***@host/stream",
      "enabled": true,
      "algorithms": [
        {
          "id": "no_helmet",
          "params": {
            "cooldown_seconds": 10,
            "min_detection_count": 3,
            "attribute_threshold": 0.5
          }
        }
      ],
      "status": {
        "online": true,
        "last_frame_age_seconds": 2,
        "fps": 24.8,
        "last_alert_at": "2026-10-23T02:15:33Z",
        "algorithm_versions": {
          "no_helmet": "2026.08.04"
        }
      }
    }
  ],
  "total": 1
}
```

说明：RTSP 密码脱敏；`status` 是运行期状态，查询本身不影响检测热路径。

## 5. 摄像头详情

### `GET /v1/cameras/{id}`

返回单个摄像头对象，结构同第 3 节；不存在返回 `404 NOT_FOUND`。

## 6. 新增摄像头

### `POST /v1/cameras`

```json
{
  "id": "1824",
  "name": "ZoneC_Outdoor_001",
  "device_code": "ICC-DEVICE-001",
  "channel_no": "1",
  "source_mode": "icc",
  "rtsp_url": "rtsp://user:pass@host/stream",
  "enabled": true,
  "algorithms": [
    {
      "id": "no_helmet",
      "params": {
        "cooldown_seconds": 10,
        "min_detection_count": 3,
        "attribute_threshold": 0.5
      }
    },
    {
      "id": "no_vest",
      "params": {
        "cooldown_seconds": 600,
        "min_detection_count": 20
      }
    }
  ]
}
```

字段规则：

- `id` 必填，实例内唯一。
- `algorithms` 必填，至少 1 个；`params` 可省略，省略时用算法默认值。
- `source_mode` 省略时：有 `rtsp_url` 视为 `direct`，只有 `device_code` + `channel_no` 视为 `icc`。
- `source_mode=direct` / `forwarded`：`rtsp_url` 必填。
- `source_mode=icc`：`device_code` + `channel_no` 必填；`rtsp_url` 可选，传了则以 `rtsp_url` 为准，否则后端按 ICC 模板拼接。
- `enabled` 默认 `true`。建议接入新摄像头时先传 `false`，`probe` 通过后再 `PATCH` 启用。

响应 `202`：

```json
{
  "operation_id": "op_20261023_0001",
  "status": "pending"
}
```

错误：

- ID 已存在：`409 CAMERA_EXISTS`
- 算法不存在：`422 ALGORITHM_NOT_FOUND`
- 参数不合法：`422 INVALID_PARAM`
- 容量不足：`409 CAPACITY_EXCEEDED`

## 7. 修改摄像头

### `PATCH /v1/cameras/{id}`

```json
{
  "name": "新名称",
  "device_code": "ICC-DEVICE-001",
  "channel_no": "2",
  "source_mode": "forwarded",
  "rtsp_url": "rtsp://...",
  "enabled": true
}
```

字段均可选；不允许修改 `id`。响应 `202`。

说明：

- `enabled:false` 会从管线移除，但保留配置，之后可以重新启用。
- 不能把最后一个 enabled 摄像头停用；请先启用另一路。
- 切换 `source_mode`/`rtsp_url` 会触发重建；建议先 `PATCH enabled:false`，改完 `probe` 通过后再启用。
- 如果把 `source_mode` 改为 `icc`，可以传 `"rtsp_url": null` 清掉旧的直连地址。

## 8. 删除摄像头

### `DELETE /v1/cameras/{id}`

响应 `202`。删除后不再拉流；历史配置保留在版本备份中。

## 9. 绑定/替换算法及参数

### `PUT /v1/cameras/{id}/algorithms`

```json
{
  "algorithms": [
    {
      "id": "no_helmet",
      "params": {
        "cooldown_seconds": 10,
        "min_detection_count": 3,
        "attribute_threshold": 0.5
      }
    },
    {
      "id": "smoking",
      "params": {
        "cooldown_seconds": 60,
        "min_detection_count": 5,
        "attribute_threshold": 0.85
      }
    }
  ]
}
```

响应 `202`。

说明：

- **全量替换**，不是追加；调用方要传完整列表。
- 至少 1 个；要停用全部算法请用 `PATCH enabled:false`。
- 解绑算法不会降低 GPU 占用，所有模型仍在对所有摄像头推理。
- 参数省略时用算法目录里的默认值；传了则只覆盖该摄像头的该算法。
- 如果新配置和当前完全一致，直接返回 `succeeded`，不触发重建。

### 9.1 参数覆盖规则

| 参数 | 覆盖范围 | 备注 |
| --- | --- | --- |
| `cooldown_seconds` | 按摄像头 | 只影响该路告警节奏 |
| `min_detection_count` | 按摄像头 | 复杂场景可调大，减少误报 |
| `attribute_threshold` | 按摄像头，仅 detector 类 | 如 `no_helmet`、`smoking` |
| `profile` | 按摄像头，仅 classifier 类；v0.1 暂未开放 | `loose` / `normal` / `strict`，下一版实现 |

约束：

- `cooldown_seconds >= 0`，`min_detection_count >= 1`，`attribute_threshold` 在 `0~1`。
- 不允许传算法目录里没有的参数。
- classifier 的 `profile` 是离散档位，不支持任意数值；每增加一个档位，后端会增加对应模型实例，GPU 开销上升。
- 参数变更会触发管线重建，全流中断时间与第 1 节一致。

## 10. 查询变更结果

### `GET /v1/operations/{id}`

```json
{
  "operation_id": "op_20261023_0001",
  "action": "create_camera",
  "camera_id": "1824",
  "status": "succeeded",
  "config_revision": 42,
  "created_at": "2026-10-23T02:15:33Z",
  "finished_at": "2026-10-23T02:16:01Z",
  "result": null,
  "error": null
}
```

`status`：

| 值 | 含义 |
| --- | --- |
| `pending` | 已受理，等待执行 |
| `applying` | 正在写配置/重启管线 |
| `succeeded` | 已生效 |
| `failed` | 执行失败，已回滚 |
| `rolled_back` | 失败后已恢复到上一版 |

`action`：`create_camera`、`update_camera`、`delete_camera`、`bind_algorithms`、`probe_camera`。

`result`：仅 `probe_camera` 使用，见第 12 节；其他操作为 `null`。

说明：operation 建议保留 7 天（可调）。

## 11. 健康检查

### `GET /healthz`

```json
{
  "status": "ok",
  "pipeline": "running",
  "config_revision": 42
}
```

不需要鉴权，供平台探活。

## 12. RTSP 探测

### `POST /v1/cameras/{id}/probe`

响应 `202`；operation 的 `result` 返回：

```json
{
  "reachable": true,
  "source_mode": "icc",
  "resolved_rtsp_url": "rtsp://user:***@host/stream",
  "resolution": "1920x1080",
  "codec": "h264",
  "fps": 25,
  "latency_ms": 320
}
```

说明：

- 只探测已配置的摄像头，不接受任意 URL，避免 SSRF。
- 推荐接入流程：`POST /v1/cameras` 时 `enabled:false` → `POST /probe` → 确认可达 → `PATCH enabled:true`。
- `resolved_rtsp_url` 会脱敏；ICC 模式下用于确认后端拼接出的地址是否正确。

## 13. Webhook（我们推给同事平台）

- 方法：`POST <同事平台回调地址>`
- Header：`Content-Type: application/json`
- Header：`X-Signature: sha256=<hex>`、`X-Timestamp: <unix_seconds>`
- 签名算法：`HMAC-SHA256(secret, f"{timestamp}.{raw_body}")`；时间戳偏差建议不超过 5 分钟。
- 响应：平台返回任意 `2xx` 视为成功；非 `2xx` 会重试。

Payload：

```json
{
  "event_id": "evt_20261023_0001",
  "schema_version": "1.0",
  "alert_type": "no_vest",
  "camera_id": "1824",
  "camera_name": "ZoneC_Outdoor_001",
  "device_code": "ICC-DEVICE-001",
  "channel_no": "1",
  "timestamp": "2026-10-23T02:15:33.12+00:00",
  "objects": [
    {
      "class": "person",
      "confidence": 0.87,
      "bbox": [840, 210, 1020, 760],
      "attributes": [
        {
          "class": "no_vest",
          "confidence": 0.5,
          "bbox": null
        }
      ]
    }
  ],
  "frame_base64": "..."
}
```

语义：

- `event_id` 在重试时保持不变，请按它去重。
- 至少一次投递，可能重复；平台侧需要幂等。
- `frame_base64` 默认携带；当前支持全局关闭（`webhook.include_frame: false`），按摄像头关闭下一版支持。
- `device_code`/`channel_no` 没配置时为 `null`。

## 14. v1 不包含

- 新算法/新模型上传、编译、上架（走内部审核）。
- classifier 类算法按摄像头传任意阈值或 `profile`（v0.1 只支持 `cooldown_seconds` / `min_detection_count`）。
- 非 RTSP 输入（GB28181/WebRTC/推流）；ICC 和转发服务最终都需提供可拉取的 RTSP 地址。
- 单帧图片推理 API（`POST /v1/infer`）。
- 多租户自助开通；本服务单租户，分发由同事平台自己做。

## 15. 请同事确认

1. 摄像头 `id` 继续用现有规则即可，还是希望 `id` 直接等于“设备编码_通道号”？我们建议 `id` 保持独立，`device_code`/`channel_no` 作为可选字段。
2. ICC 的 RTSP 地址拼接模板、认证方式能否提供？如果后端不拼，是否统一走转发服务的 `rtsp_url`？
3. classifier 的严格程度用 `loose`/`normal`/`strict` 三档是否够？如果不够，需要几档，是否接受 GPU 开销上升？
4. Webhook 签名密钥怎么分发？`frame_base64` 是否有大小上限或关闭需求？
5. 32 路是引擎硬上限，实际单机计划接多少路？是否需要预留扩容空间？
6. `status.online` 按“最近 30 秒内有帧”判定是否合适？`last_alert_at` 默认持久化，是否还需要按算法拆分“每个算法最后告警时间”？
7. 状态查询频率大概多少（用于评估轮询压力）？
8. `probe` 结果里的 `resolution`/`codec`/`fps`/`latency_ms` 是否够用？

## 16. 变更记录

| 日期 | 版本 | 变更 |
| --- | --- | --- |
| 2026-10-23 | v1-draft | 初稿 |
| 2026-10-23 | v1.1-draft | 根据反馈：probe 进 v1；增加 `device_code`/`channel_no`、ICC/转发源、按摄像头参数、状态字段、Webhook 签名与 `event_id`；明确单租户 |
