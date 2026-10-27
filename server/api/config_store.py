"""config.yaml 的读写、校验和对外视图。

控制面只改配置，不直接碰 DeepStream 管线：
- 摄像头/算法参数写入 config.yaml；
- 生成 __cam__ 规则实现“按摄像头参数”，检测进程按普通规则读取；
- 写入前复用 server.config.validate_config，失败不落盘；
- 每次写入保留历史版本，供失败回滚。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import yaml

from server.config import validate_config
from server.alert.regions import parse_zones
from server.model_spec import KIND_CLASSIFIER, KIND_DETECTOR, parse_gies
from server.api.errors import ApiError
from server.api.util import atomic_write_json, atomic_write_text, mask_rtsp_url, now_iso

GENERATED_PREFIX = "__cam__"
CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ALLOWED_PARAMS = ("cooldown_seconds", "min_detection_count", "attribute_threshold",
                  "zones", "vacant_frames")
MAX_HISTORY = 20
MAX_CAMERAS = 32  # 与 tools/model_build.py 的引擎 max_batch 一致


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _canonical_zones(raw) -> list[dict]:
    """区域统一写成契约形态 {id, anchor, points: [[x, y], ...]}；平台 RuleId/Points 形态也接受。"""
    return [
        {"id": z.id, "anchor": z.anchor, "points": [[x, y] for x, y in z.points]}
        for z in parse_zones(raw, where="zones")
    ]


def _locked(method):
    """把“读配置 → 改内存 → 校验”整段串行化，避免并发请求互相覆盖。"""
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class ConfigStore:
    """config.yaml 的单一写入口。"""

    def __init__(self, config_path: str | Path, runtime_dir: str | Path):
        self.config_path = Path(config_path)
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "config-history").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- 读
    def read(self) -> dict:
        if not self.config_path.exists():
            raise ApiError(500, "CONFIG_MISSING", f"配置文件不存在: {self.config_path}")
        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise ApiError(500, "CONFIG_INVALID", f"配置文件无法解析: {exc}") from exc
        if not isinstance(raw, dict):
            raise ApiError(500, "CONFIG_INVALID", "配置文件顶层必须是对象")
        return raw

    def revision(self) -> int:
        path = self.runtime_dir / "revision.json"
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["revision"])
        except Exception:
            return 0

    # ---------------------------------------------------------------- 写
    def backup(self, raw: dict) -> Path:
        with self._lock:
            name = f"{time.strftime('%Y%m%d-%H%M%S')}-rev{self.revision()}.yaml"
            path = self.runtime_dir / "config-history" / name
            atomic_write_text(path, yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
            files = sorted((self.runtime_dir / "config-history").glob("*.yaml"))
            for old in files[:-MAX_HISTORY]:
                try:
                    old.unlink()
                except OSError:
                    pass
            return path

    def write_raw(self, raw: dict) -> None:
        """校验并原子替换配置文件。"""
        with self._lock:
            try:
                validate_config(raw, self.config_path)
            except ValueError as exc:
                raise ApiError(400, "CONFIG_INVALID", str(exc)) from exc

            text = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
            tmp = self.config_path.with_name(self.config_path.name + ".tmp")
            try:
                atomic_write_text(tmp, text)
                os.replace(tmp, self.config_path)
            except OSError:
                # 单文件 bind mount 不允许 rename 时退回原地写；仍保留 .bak。
                bak = self.config_path.with_name(self.config_path.name + ".bak")
                try:
                    atomic_write_text(bak, self.config_path.read_text(encoding="utf-8"))
                except OSError:
                    pass
                self.config_path.write_text(text, encoding="utf-8")
                try:
                    tmp.unlink()
                except OSError:
                    pass
            try:
                os.chmod(self.config_path, 0o600)
            except OSError:
                pass

    def bump_revision(self) -> int:
        with self._lock:
            revision = self.revision() + 1
            atomic_write_json(self.runtime_dir / "revision.json", {"revision": revision})
            return revision

    def request_reload(self, revision: int) -> None:
        atomic_write_json(
            self.runtime_dir / "reload.request",
            {"revision": revision, "requested_at": now_iso()},
        )

    def activate(self, candidate: dict) -> int:
        """备份当前配置 → 写入新配置 → 递增版本 → 通知检测进程重载。"""
        with self._lock:
            self.backup(self.read())
            self.write_raw(candidate)
            revision = self.bump_revision()
            self.request_reload(revision)
            return revision

    def rollback(self, backup_path: Path) -> int:
        with self._lock:
            raw = yaml.safe_load(Path(backup_path).read_text(encoding="utf-8")) or {}
            self.write_raw(raw)
            revision = self.bump_revision()
            self.request_reload(revision)
            return revision

    # ---------------------------------------------------------------- 对外视图
    def list_algorithms(self) -> list[dict]:
        raw = self.read()
        gies = parse_gies((raw.get("model") or {}).get("gies"))
        generated = self._generated_map(raw)
        items = []
        for name, rule in (raw.get("rules") or {}).items():
            if name in generated:
                continue
            gie = gies.get((rule or {}).get("gie"))
            if gie is None:
                continue
            schema = {
                "cooldown_seconds": {
                    "type": "number", "min": 0,
                    "default": rule.get("cooldown_seconds", 0),
                },
                "min_detection_count": {
                    "type": "integer", "min": 1,
                    "default": rule.get("min_detection_count", 1),
                },
            }
            gie_kind = gie.kind if gie is not None else None
            presence = (rule or {}).get("trigger") == "presence"
            if gie_kind == KIND_DETECTOR and not presence:
                schema["attribute_threshold"] = {
                    "type": "number", "min": 0, "max": 1,
                    "default": rule.get("attribute_threshold", 0.5),
                }
            elif gie_kind == KIND_CLASSIFIER:
                schema["profile"] = {
                    "type": "string",
                    "enum": ["loose", "normal", "strict"],
                    "default": "normal",
                    "supported": False,
                    "note": "v0.2 仅支持全局配置；按摄像头阈值需要下一版",
                }
            # 区域能力：省略 zones = 全图生效；presence 型（区域入侵）必须给出
            schema["zones"] = {
                "type": "array",
                "items": {
                    "id": "string",
                    "anchor": {"enum": ["feet", "center"], "default": "feet"},
                    "points": "array of [x, y], 归一化 0~1, >=3 顶点",
                },
                "required": (rule or {}).get("trigger") == "presence",
                "note": "省略则本算法在全图判定；给出则仅区域内命中才告警",
            }
            if (rule or {}).get("trigger") == "presence":
                schema["vacant_frames"] = {
                    "type": "integer", "min": 1,
                    "default": rule.get("vacant_frames", 1),
                    "note": "离开去抖：区域内连续 N 帧无人才算本次事件结束（按该路处理帧率换算）",
                }
            items.append({
                "id": name,
                "display_name": rule.get("display_name") or name,
                "kind": gie.kind,
                "model_version": gie.version,
                "params_schema": schema,
            })
        return sorted(items, key=lambda item: item["id"])

    def list_cameras(self, status_by_id: dict | None = None) -> list[dict]:
        raw = self.read()
        status_by_id = status_by_id or {}
        return [
            self._camera_view(raw, cam, status_by_id)
            for cam in (raw.get("cameras") or [])
        ]

    def get_camera(self, camera_id: str, status_by_id: dict | None = None) -> dict:
        raw = self.read()
        cam = self._find_camera(raw, camera_id)
        return self._camera_view(raw, cam, status_by_id or {})

    def get_camera_raw(self, camera_id: str) -> dict:
        return copy.deepcopy(self._find_camera(self.read(), camera_id))

    # ---------------------------------------------------------------- 变更
    @_locked
    def create_camera(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ApiError(400, "INVALID_ARGUMENT", "请求体必须是 JSON 对象")
        raw = self.read()
        camera_id = payload.get("id")
        self._validate_camera_id(camera_id)
        if any(str(cam.get("id")) == str(camera_id) for cam in (raw.get("cameras") or [])):
            raise ApiError(409, "CAMERA_EXISTS", f"camera {camera_id} already exists")

        cam = self._camera_from_payload(payload)
        raw.setdefault("cameras", []).append(cam)
        self._set_algorithms(raw, cam, payload.get("algorithms"))
        self._validate_candidate(raw)
        return raw

    @_locked
    def update_camera(self, camera_id: str, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ApiError(400, "INVALID_ARGUMENT", "请求体必须是 JSON 对象")
        if "algorithms" in payload:
            raise ApiError(400, "INVALID_ARGUMENT", "修改算法请用 PUT /v1/cameras/{id}/algorithms")
        raw = self.read()
        cam = self._find_camera(raw, camera_id)
        old_mode = cam.get("source_mode")
        if "id" in payload and str(payload["id"]) != str(camera_id):
            raise ApiError(400, "INVALID_ARGUMENT", "不允许修改摄像头 id")
        self._validate_camera_fields(payload)

        for key in ("name", "device_code", "channel_no", "source_mode", "rtsp_url", "enabled"):
            if key in payload:
                cam[key] = payload[key]

        # ICC 模式下改了设备/通道但没显式给 URL 时，清掉旧 URL 让后端按模板重拼；
        # 切到直连/转发模式则必须同时提供新的 rtsp_url。
        target_mode = payload.get("source_mode", old_mode)
        source_changed = any(k in payload for k in ("source_mode", "device_code", "channel_no"))
        if source_changed and target_mode == "icc" and "rtsp_url" not in payload:
            cam["rtsp_url"] = None
        elif ("source_mode" in payload
              and payload["source_mode"] != old_mode
              and "rtsp_url" not in payload
              and target_mode != "icc"
              and not cam.get("rtsp_url")):
            raise ApiError(400, "INVALID_ARGUMENT", "切换 source_mode 时请同时提供 rtsp_url")

        self._normalize_source(cam)
        self._validate_candidate(raw)
        return raw

    @_locked
    def delete_camera(self, camera_id: str) -> dict:
        raw = self.read()
        cam = self._find_camera(raw, camera_id)
        enabled = [c for c in (raw.get("cameras") or []) if c.get("enabled", True)]
        if cam.get("enabled", True) and len(enabled) <= 1:
            raise ApiError(409, "LAST_CAMERA", "至少保留一个 enabled 摄像头；请先启用另一路")
        raw["cameras"] = [c for c in raw.get("cameras") or [] if str(c.get("id")) != str(camera_id)]
        self._remove_generated_rules(raw, camera_id)
        self._validate_candidate(raw)
        return raw

    @_locked
    def set_algorithms(self, camera_id: str, payload) -> dict:
        if isinstance(payload, dict):
            algorithms = payload.get("algorithms")
        else:
            algorithms = payload
        raw = self.read()
        cam = self._find_camera(raw, camera_id)
        self._set_algorithms(raw, cam, algorithms)
        self._validate_candidate(raw)
        return raw

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _generated_map(raw: dict) -> dict:
        meta = raw.get("api_meta") or {}
        mapping = meta.get("generated_rules")
        return mapping if isinstance(mapping, dict) else {}

    @staticmethod
    def _find_camera(raw: dict, camera_id: str) -> dict:
        for cam in raw.get("cameras") or []:
            if str(cam.get("id")) == str(camera_id):
                return cam
        raise ApiError(404, "NOT_FOUND", f"camera {camera_id} not found")

    @staticmethod
    def _validate_camera_id(camera_id) -> None:
        if not isinstance(camera_id, str) or not CAMERA_ID_RE.match(camera_id):
            raise ApiError(
                400, "INVALID_ARGUMENT",
                "id 必须是 1~64 位字母、数字、下划线或短横线",
            )

    @staticmethod
    def _validate_camera_fields(payload: dict) -> None:
        checks = {
            "name": str,
            "enabled": bool,
            "device_code": str,
            "channel_no": str,
            "source_mode": str,
            "rtsp_url": str,
        }
        for key, expected in checks.items():
            if key not in payload or payload[key] is None:
                continue
            if not isinstance(payload[key], expected):
                raise ApiError(400, "INVALID_ARGUMENT", f"{key} 类型不正确")

    def _camera_from_payload(self, payload: dict) -> dict:
        camera_id = payload.get("id")
        self._validate_camera_id(camera_id)
        self._validate_camera_fields(payload)
        cam = {
            "id": camera_id,
            "name": payload.get("name") or camera_id,
            "enabled": True if payload.get("enabled") is None else bool(payload.get("enabled")),
            "device_code": payload.get("device_code"),
            "channel_no": payload.get("channel_no"),
            "source_mode": payload.get("source_mode"),
            "rtsp_url": payload.get("rtsp_url"),
        }
        self._normalize_source(cam)
        return cam

    @staticmethod
    def _normalize_source(cam: dict) -> None:
        mode = cam.get("source_mode")
        if not mode:
            if not cam.get("rtsp_url") and cam.get("device_code") and cam.get("channel_no"):
                mode = "icc"
            else:
                mode = "direct"
        if mode not in ("direct", "icc", "forwarded"):
            raise ApiError(400, "INVALID_ARGUMENT", "source_mode 必须是 direct/icc/forwarded")
        cam["source_mode"] = mode

        if mode == "icc":
            if not cam.get("device_code") or not cam.get("channel_no"):
                raise ApiError(400, "INVALID_ARGUMENT", "source_mode=icc 需要 device_code 和 channel_no")
            if not cam.get("rtsp_url"):
                template = os.environ.get("SAFETY_ICC_RTSP_TEMPLATE")
                if not template:
                    raise ApiError(
                        400, "ICC_TEMPLATE_MISSING",
                        "未配置 SAFETY_ICC_RTSP_TEMPLATE，请直接提供 rtsp_url 或配置 ICC 模板",
                    )
                try:
                    cam["rtsp_url"] = template.format(
                        device_code=cam["device_code"],
                        channel_no=cam["channel_no"],
                        camera_id=cam.get("id", ""),
                    )
                except KeyError as exc:
                    raise ApiError(500, "ICC_TEMPLATE_INVALID", f"ICC 模板变量不存在: {exc}") from exc
        elif not cam.get("rtsp_url"):
            raise ApiError(400, "INVALID_ARGUMENT", "source_mode=direct/forwarded 需要 rtsp_url")
        elif not str(cam["rtsp_url"]).startswith("rtsp://"):
            raise ApiError(400, "INVALID_ARGUMENT", "rtsp_url 必须以 rtsp:// 开头")

    def _set_algorithms(self, raw: dict, cam: dict, algorithms) -> None:
        if not isinstance(algorithms, list) or not algorithms:
            raise ApiError(400, "INVALID_ARGUMENT", "algorithms 必须是非空列表")

        rules = raw.setdefault("rules", {})
        gies = parse_gies((raw.get("model") or {}).get("gies"))
        self._remove_generated_rules(raw, str(cam["id"]))
        meta = raw.setdefault("api_meta", {}).setdefault("generated_rules", {})
        active = []
        seen = set()

        for item in algorithms:
            if isinstance(item, str):
                alg_id, params = item, {}
            elif isinstance(item, dict):
                alg_id = item.get("id")
                params = item.get("params") or {}
            else:
                raise ApiError(400, "INVALID_ARGUMENT", "algorithms 每项必须是字符串或对象")

            if not isinstance(alg_id, str) or not alg_id:
                raise ApiError(400, "INVALID_ARGUMENT", "algorithms[].id 必须是非空字符串")
            if alg_id in seen:
                raise ApiError(400, "INVALID_ARGUMENT", f"算法重复绑定: {alg_id}")
            seen.add(alg_id)
            base = rules.get(alg_id)
            if not isinstance(base, dict) or alg_id in self._generated_map(raw):
                raise ApiError(422, "ALGORITHM_NOT_FOUND", f"算法不存在: {alg_id}")
            gie = gies.get(base.get("gie"))
            if gie is None:
                raise ApiError(422, "ALGORITHM_NOT_FOUND", f"算法 {alg_id} 引用的模型不存在")
            if not isinstance(params, dict):
                raise ApiError(400, "INVALID_ARGUMENT", "params 必须是对象")

            self._validate_params(alg_id, gie.kind, params,
                                  trigger=(base.get("trigger") or "violation"))
            if not params:
                active.append(alg_id)
                continue
            if "zones" in params:
                params = {**params, "zones": _canonical_zones(params["zones"])}

            token = _safe_token(str(cam["id"]))
            digest = hashlib.sha1(str(cam["id"]).encode("utf-8")).hexdigest()[:8]
            gen_name = f"{GENERATED_PREFIX}{token}__{digest}__{alg_id}"
            rule = copy.deepcopy(base)
            rule.update(params)
            rule["display_name"] = base.get("display_name") or alg_id
            rule["alert_type"] = base.get("alert_type") or alg_id
            rules[gen_name] = rule
            meta[gen_name] = {"camera_id": str(cam["id"]), "algorithm": alg_id}
            active.append(gen_name)

        cam["active_rules"] = active

    @staticmethod
    def _validate_params(alg_id: str, kind: str, params: dict,
                         trigger: str = "violation") -> None:
        unknown = set(params) - set(ALLOWED_PARAMS) - {"profile"}
        if unknown:
            raise ApiError(422, "INVALID_PARAM", f"算法 {alg_id} 不支持参数: {sorted(unknown)}")
        # presence 型（区域入侵）只看「区域内有没有人」，不做任何模型置信度判定；
        # 它的 attribute_threshold 只是校验器要求的占位值，不对外开放，避免平台误调。
        if trigger == "presence" and "attribute_threshold" in params:
            raise ApiError(
                422, "UNSUPPORTED_PARAM",
                f"算法 {alg_id} 是区域存在性判定（presence），不支持 attribute_threshold",
            )
        if "profile" in params:
            raise ApiError(
                422, "UNSUPPORTED_PARAM",
                f"算法 {alg_id} 的 profile 参数暂未在 v0.2 开放；"
                "请用 cooldown_seconds/min_detection_count，或等下一版",
            )
        # 区域：结构与取值复用运行期同一套解析（server/alert/regions.parse_zones），
        # 避免「API 收得下、启动时炸」的双标；坏区域在写入前就 422 回去。
        if "zones" in params:
            try:
                parse_zones(params["zones"], where=f"api.{alg_id}")
            except ValueError as exc:
                raise ApiError(422, "INVALID_PARAM", str(exc)) from exc
        if "vacant_frames" in params:
            value = params["vacant_frames"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ApiError(422, "INVALID_PARAM", "vacant_frames 必须是 >=1 的整数")
        if "cooldown_seconds" in params:
            value = params["cooldown_seconds"]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ApiError(422, "INVALID_PARAM", "cooldown_seconds 必须是非负数")
        if "min_detection_count" in params:
            value = params["min_detection_count"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ApiError(422, "INVALID_PARAM", "min_detection_count 必须是 >=1 的整数")
        if "attribute_threshold" in params:
            if kind != KIND_DETECTOR:
                raise ApiError(
                    422, "UNSUPPORTED_PARAM",
                    f"算法 {alg_id} 是 classifier，按摄像头 attribute_threshold 暂未开放",
                )
            value = params["attribute_threshold"]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 <= value <= 1):
                raise ApiError(422, "INVALID_PARAM", "attribute_threshold 必须在 0~1")

    @staticmethod
    def _remove_generated_rules(raw: dict, camera_id: str) -> None:
        meta = raw.setdefault("api_meta", {}).setdefault("generated_rules", {})
        rules = raw.setdefault("rules", {})
        for name, info in list(meta.items()):
            if str((info or {}).get("camera_id")) == str(camera_id):
                rules.pop(name, None)
                meta.pop(name, None)

    def _camera_view(self, raw: dict, cam: dict, status_by_id: dict) -> dict:
        gies = parse_gies((raw.get("model") or {}).get("gies"))
        generated = self._generated_map(raw)
        rules = raw.get("rules") or {}
        algorithms = []
        versions = {}

        active = cam.get("active_rules")
        if active is None:
            active = [name for name in rules if name not in generated]

        for rule_name in active:
            info = generated.get(rule_name) or {}
            alg_id = info.get("algorithm") or rule_name
            rule = rules.get(rule_name) or {}
            gie = gies.get(rule.get("gie"))
            params = {}
            if "cooldown_seconds" in rule:
                params["cooldown_seconds"] = rule["cooldown_seconds"]
            if "min_detection_count" in rule:
                params["min_detection_count"] = rule["min_detection_count"]
            if (gie is not None and gie.kind == KIND_DETECTOR
                    and rule.get("trigger") != "presence"
                    and "attribute_threshold" in rule):
                params["attribute_threshold"] = rule["attribute_threshold"]
            # 区域参数必须回显：视图是「读→改→写」闭环的唯一事实来源，
            # 漏掉会导致下一次按算法写入把已有 zones 静默清空。
            if "zones" in rule:
                params["zones"] = copy.deepcopy(rule["zones"])
            if "vacant_frames" in rule:
                params["vacant_frames"] = rule["vacant_frames"]
            algorithms.append({"id": alg_id, "params": params})
            if gie is not None:
                versions[alg_id] = gie.version

        status = status_by_id.get(str(cam.get("id"))) or {}
        return {
            "id": cam.get("id"),
            "name": cam.get("name", cam.get("id")),
            "device_code": cam.get("device_code"),
            "channel_no": cam.get("channel_no"),
            "source_mode": cam.get("source_mode", "direct"),
            "rtsp_url": mask_rtsp_url(cam.get("rtsp_url")),
            "enabled": bool(cam.get("enabled", True)),
            "algorithms": algorithms,
            "status": {
                "online": bool(status.get("online", False)),
                "last_frame_age_seconds": status.get("last_frame_age_seconds"),
                "fps": status.get("fps", 0),
                "last_alert_at": status.get("last_alert_at"),
                "algorithm_versions": versions,
            },
        }

    @staticmethod
    def _validate_candidate(raw: dict) -> None:
        cameras = raw.get("cameras") or []
        ids = [str(cam.get("id")) for cam in cameras]
        if len(ids) != len(set(ids)):
            raise ApiError(400, "INVALID_ARGUMENT", "摄像头 id 不能重复")
        enabled = [cam for cam in cameras if cam.get("enabled", True)]
        if len(enabled) > MAX_CAMERAS:
            raise ApiError(
                409, "CAPACITY_EXCEEDED",
                f"最多 {MAX_CAMERAS} 路 enabled 摄像头，当前 {len(enabled)} 路",
            )
        try:
            validate_config(raw, Path("<candidate>"))
        except ValueError as exc:
            raise ApiError(400, "CONFIG_INVALID", str(exc)) from exc
