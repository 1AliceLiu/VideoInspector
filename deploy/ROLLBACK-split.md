# 双 GPU 双实例（split profile）切换 / 回退手册

> 2026-11-09。改动只涉及 `deploy/docker-compose.yml`（新增 detector-a/b）与
> `deploy/config.a.yaml`、`config.b.yaml`（新增）。**旧 `detector` 服务原样保留**，
> 当前生产仍由它单实例运行，本次改配置未重启/未影响生产线。

## 改动快照

| 文件 | 说明 |
| :--- | :--- |
| `docker-compose.yml` | 新增 `detector-a`/`detector-b`，`profiles: ["split"]`，默认不启动 |
| `docker-compose.yml.pre-split.bak-20261109` | 改动前 compose 备份（回退可直接覆盖） |
| `config.a.yaml` | A 侧 22 路（GPU0），helmet=1107-update，日志 logs/server-a.log |
| `config.b.yaml` | B 侧 22 路（GPU1），helmet=1107-update，日志 logs/server-b.log |

摄像头划分（互斥，44 路全量）：
- A：1195,1385,1439,1731,1825,1918,2172,2201,2628,2633,2635,2644,2666,2668,2674,2676,2741,353,500,671,762,891
- B：1096,1168,1363,1384,1485,1824,1912,1924,2168,2634,2636,2639,2667,2669,2675,2740,2742,496,670,676,720,880

## 切换（需要停机窗口，预计 1–2 分钟）

```bash
cd /home/pzq/projects/VideoInspector/deploy
docker compose stop detector                # 停旧单实例
docker compose --profile split up -d detector-a
sleep 30                                    # 等 A 起满 22 路、写完 labels/INI
docker compose --profile split up -d detector-b
docker compose ps                           # 应为 detector-a / detector-b 两个 Up
```

验证：
- 两容器日志 `startup camera=` 合计 44；`docker logs <name> | grep -c startup` 各 22；
- `nvidia-smi`：GPU0/GPU1 各有一个 python3，util 有负载；
- 生产 FPS 基线 12.4–12.5/路；
- 平台告警正常（webhook 成功行），无重复摄像头。

## 回退（1–2 分钟，零重建）

```bash
cd /home/pzq/projects/VideoInspector/deploy
docker compose stop detector-a detector-b
docker compose up -d detector              # 旧单实例（config.yaml 未改动）
docker compose ps                          # 应为 detector Up
```

如需连 compose 文件一起回退：

```bash
cp docker-compose.yml.pre-split.bak-20261109 docker-compose.yml
```

## 注意

- **必须带 service 名启动**：`docker compose --profile split up -d` 不带服务名会同时拉起旧
  `detector`，造成同一摄像头被两个实例拉流/重复告警。统一用 `... up -d detector-a detector-b`。
- `privileged: true` 会暴露全部 GPU；计算卡由 `CUDA_VISIBLE_DEVICES=0/1` 钉住，
  `device_ids` 在此模式下无效（灰度验证时踩过坑）。
- `generated/` 两实例共享（运行期只读）；启动间隔 20–30s，避免同时重写 labels/INI。
- 预览输出当前未启用；若启用，需给 A/B 分配不同宿主端口（当前 compose 未映射端口）。
- 日志：`deploy/logs/server-a.log`、`server-b.log`（旧实例是 `server.log`）。
- 单卡 44 路 @960 实测会掉 ~20% FPS 且峰值 87%，已判定不可行；本方案是双卡 22/22。
