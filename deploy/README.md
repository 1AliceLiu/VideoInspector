
## 浸泡测试

cd deploy 

docker compose --profile soak down
docker compose --profile soak up -d --build
               └──────┬────────┘ └┬┘ └──┬──┘
               激活soak组(拉起接收端)  后台  首次顺便构建镜像

## 部署

docker compose up -d --build --no-cache

docker compose up -d
docker compose down

## B2 聚合 API（A/B 双实例，可写）

首次上线需要重建 detector-a/b，把各自的 runtime 挂到宿主机：

```bash
cd deploy
cp config.a.yaml config.a.yaml.bak.$(date +%Y%m%d_%H%M%S)
cp config.b.yaml config.b.yaml.bak.$(date +%Y%m%d_%H%M%S)
docker compose stop detector-a
docker compose --profile split up -d --build detector-a
sleep 30
docker compose stop detector-b
docker compose --profile split up -d --build detector-b
docker compose --profile split up -d --build api
docker compose ps
```

先确保 `SAFETY_API_KEY` 已设置（compose 从当前 shell 或 `deploy/.env` 读取）。
API 只监听宿主机回环地址 `127.0.0.1:18083`，通过现有 nginx 18081 的
`/video-inspector/` 反向代理对外：

```text
http://118.140.130.26:18081/video-inspector/v1
```

nginx 片段（放在 18081 站点 server 内，静态 `location /` 之前）：

```nginx
location ^~ /video-inspector/ {
    proxy_pass http://127.0.0.1:18083/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    client_max_body_size 2m;
    proxy_read_timeout 120s;
}
```

改完执行 `nginx -t && nginx -s reload`。

本地验证：

```bash
curl -H "Authorization: Bearer $SAFETY_API_KEY" \
  http://127.0.0.1:18083/v1/cameras/1195
```

B2 写路径：

- `GET /v1/cameras` 合并 A/B，摄像头带 `node` 和 `algorithm_count`
- `POST /v1/cameras/{id}/algorithms` 追加算法，只改目标节点
- `POST /v1/cameras` 新建摄像头；`node` 可选，省略时自动放路数较少的一侧
- 写配置会经 `runtime-a/b/reload.request` 触发对应 detector 重载
- 全局同一时刻只允许一个变更，A/B 写操作串行

手动验证给摄像头追加算法：

```bash
curl -X POST -H "Authorization: Bearer $SAFETY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"algorithms":[{"id":"smoking"}]}' \
  http://127.0.0.1:18083/v1/cameras/1195/algorithms
```

然后轮询 operation：

```bash
curl -H "Authorization: Bearer $SAFETY_API_KEY" \
  http://127.0.0.1:18083/v1/operations/<operation_id>
```

> 注意：`config.a.yaml` / `config.b.yaml` 是单文件 bind mount，API 写入时
> `os.replace` 可能失败，`ConfigStore.write_raw()` 会回退为原地写。上线前先用
> 一个非关键摄像头验证一次追加，确认宿主机 YAML 真的发生变化。
