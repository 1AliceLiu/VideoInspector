# 方案提案：吸烟告警的大模型二级复核（四宫格连拍 + yes/no 门控）

> 状态：**提案，未实施**。主工程零改动；批准后按下方步骤合入。

## 思路（对应讨论结论）

1. 现有 `smoke` 小模型继续在 DeepStream 链里**实时初筛**（建议把 `rules.smoking.attribute_threshold` 从 0.75 降到 ~0.5，放宽到宁可误报）。
2. 状态机触发告警的**瞬间**起，从证据帧缓存连拍采样 4 张新帧（间隔 `sample_ms=120`，上限 `deadline_ms=1500`），按触发帧的 person 框（外扩 15%）裁剪。
3. 4 张裁剪拼成 **2×2 宫格图**（512×512，一次调用固定开销只付一遍，比调 4 次快 ~3 倍），发给 OpenAI 兼容接口（vLLM 部署的 Qwen2.5-VL，`max_tokens=1`）问：「此人是否在吸烟，只答 yes/no」。
4. 判定门控 webhook（灰度期先 `shadow: true` 只观察，见下）：
   - `yes` → 正常推送，payload 附 `vlm: {verdict, model, frames}`
   - `no`  → 丢弃，只记日志
   - 大模型不可用 / 采不到帧 → `fail_open=true` 照常推送（`verdict=skipped`），宁误报不漏报
5. 端到端告警延迟 ≈ 采样 0.4s + 推理 0.15s < 1s；采样与 HTTP 全在告警 daemon 线程，**不阻塞检测流线程**。每条复核记一行日志 `vlm-review ... verdict=... sample=Nms infer=Nms total=Nms`（total=触发线程启动→大模型判定返回的全链路耗时）。

大模型服务部署在空闲的 GPU 3（全机共享一个 vLLM 容器，权重已放 `/home/lsb/model/`；
注意宿主机 8010/8011/8012 已被 easyocr 系列容器占用，故 vLLM 用 **8013**）：

```bash
docker run -d --gpus '"device=3"' --network host --name vllm --restart unless-stopped \
  -v /home/lsb/model:/model vllm/vllm-openai:latest \
  --model /model/Qwen2.5-VL-7B-Instruct-AWQ --port 8013
# 验证：curl http://127.0.0.1:8013/v1/models
```

## 本目录文件

| 文件 | 说明 |
| :--- | :--- |
| `vlm_review.py` | 新模块 → 放到 `server/alert/`（连拍采样 / 裁剪 / 宫格 / 大模型问答） |
| `test_vlm_review.py` | 新测试 → 放到 `tests/`（本机假 HTTP 服务模拟 vLLM，6 用例） |
| `existing-files.patch` | 对既有 4 处代码的改动补丁（见下） |

补丁覆盖的既有改动：
- `server/alert/manager.py`：触发后按 `vlm.rules` 分流到复核线程；payload 附判定
- `server/pipeline/builder.py`：把 vlm 配置 / 证据 frame_cache / source_id 注入 AlertManager
- `server/config.py`：启动校验（enabled 但缺 url、rules 引用不存在的规则 → fail fast）
- `deploy/config.yaml`：**需手工粘贴** `vlm:` 配置节（放在 `webhook:` 节之后）：

```yaml
vlm:
  enabled: true
  shadow: true        # 观察模式：判定只进日志，不拦告警、payload 不附字段
  url: http://vllm:8013/v1/chat/completions   # 检测容器走该 DNS 名；宿主机 curl 测试用 127.0.0.1:8013
  model: /model/Qwen2.5-VL-7B-Instruct-AWQ
  rules: [smoking]
  frames: 4
  sample_ms: 120
  deadline_ms: 1500
  pad: 0.15           # person bbox 外扩比例，烟头出框时调大（旁边有人时慎调太大）
  fail_open: true
```

## 合入步骤（批准后）

```bash
cd /home/pzq/projects/VideoInspector
git apply proposals/vlm-review/existing-files.patch
cp proposals/vlm-review/vlm_review.py server/alert/
cp proposals/vlm-review/test_vlm_review.py tests/
# 再把上面 vlm: 节粘进 deploy/config.yaml（enabled: true + shadow: true 起步，只观察）
# 容器内验证（需 cv2/numpy/simplejpeg；先 mkdir，docker cp 目标不存在会拷错结构）:
docker exec safety-detector mkdir -p /tmp/vi-check && \
docker cp server safety-detector:/tmp/vi-check/server && \
docker cp tests safety-detector:/tmp/vi-check/tests && \
docker exec -w /tmp/vi-check safety-detector python3 tests/test_vlm_review.py
```

## 已知简化（ponytail 声明）

- 连拍裁剪沿用**触发帧的 bbox**（未跟踪）：0.5s 内位移小；烟头常出框则先调大 `pad`（默认 0.15 暂不加，按灰度日志定夺），仍不行再接 nvtracker。
- 灰度期 `shadow: true`：大模型只观察不拦截，核对一周日志 `vlm-review ... verdict=` 的对错与 `total=` 耗时后，再关 shadow 让它拦告警（届时 payload 才开始附 `vlm` 字段）。
