"""多节点（A/B）聚合：合并配置视图、运行状态，校验一致性并路由写操作。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from server.api.config_store import ConfigStore
from server.api.errors import ApiError

DEFAULT_MAX_CAMERAS_PER_NODE = 32


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass(frozen=True)
class Node:
    name: str
    store: ConfigStore
    status_path: Path


class NodeRegistry:
    """把多个单配置文件 ConfigStore 聚合成一个对外视图。"""

    def __init__(self, nodes: dict[str, Node],
                 max_cameras_per_node: int = DEFAULT_MAX_CAMERAS_PER_NODE):
        if not nodes:
            raise ValueError("至少需要一个节点")
        self.nodes = dict(nodes)
        self.max_cameras_per_node = max_cameras_per_node

    @classmethod
    def from_paths(cls, paths: dict[str, tuple[str | Path, str | Path]],
                   max_cameras_per_node: int = DEFAULT_MAX_CAMERAS_PER_NODE):
        nodes = {}
        for name, (config_path, runtime_dir) in paths.items():
            runtime = Path(runtime_dir)
            nodes[name] = Node(
                name=name,
                store=ConfigStore(config_path, runtime),
                status_path=runtime / "status.json",
            )
        return cls(nodes, max_cameras_per_node)

    def validate(self) -> None:
        """启动校验：节点容量、跨节点重复 camera id、算法目录一致。"""
        seen: dict[str, str] = {}
        for name, node in self.nodes.items():
            cameras = node.store.list_cameras({})
            enabled = [cam for cam in cameras if cam.get("enabled", True)]
            if len(enabled) > self.max_cameras_per_node:
                raise ValueError(
                    f"节点 {name} 启用摄像头 {len(enabled)} 路，超过上限 "
                    f"{self.max_cameras_per_node}"
                )
            for cam in cameras:
                camera_id = str(cam.get("id"))
                if camera_id in seen:
                    raise ValueError(
                        f"摄像头 {camera_id} 同时出现在节点 {seen[camera_id]} 和 {name}"
                    )
                seen[camera_id] = name

        algorithms = self.list_algorithms()
        if not algorithms:
            raise ValueError("算法目录为空")

    def status_by_id(self) -> dict[str, dict]:
        status: dict[str, dict] = {}
        sources: dict[str, str] = {}
        for name, node in self.nodes.items():
            data = _read_json(node.status_path)
            for cam in data.get("cameras") or []:
                camera_id = str(cam.get("id"))
                if camera_id in status:
                    raise ApiError(
                        500, "DUPLICATE_CAMERA",
                        f"摄像头 {camera_id} 的状态同时出现在节点 "
                        f"{sources[camera_id]} 和 {name}",
                    )
                status[camera_id] = cam
                sources[camera_id] = name
        return status

    def list_cameras(self) -> list[dict]:
        status = self.status_by_id()
        items: list[dict] = []
        seen: dict[str, str] = {}
        for name, node in self.nodes.items():
            for cam in node.store.list_cameras(status):
                camera_id = str(cam.get("id"))
                if camera_id in seen:
                    raise ApiError(
                        500, "DUPLICATE_CAMERA",
                        f"摄像头 {camera_id} 同时出现在节点 {seen[camera_id]} 和 {name}",
                    )
                seen[camera_id] = name
                cam["node"] = name
                items.append(cam)
        return items

    def get_camera(self, camera_id: str) -> dict:
        status = self.status_by_id()
        found: list[tuple[str, dict]] = []
        for name, node in self.nodes.items():
            try:
                found.append((name, node.store.get_camera(camera_id, status)))
            except ApiError as exc:
                if exc.status != 404:
                    raise
        if not found:
            raise ApiError(404, "NOT_FOUND", f"camera {camera_id} not found")
        if len(found) > 1:
            names = ", ".join(name for name, _ in found)
            raise ApiError(
                409, "DUPLICATE_CAMERA",
                f"摄像头 {camera_id} 同时存在于节点 {names}",
            )
        name, cam = found[0]
        cam["node"] = name
        return cam

    def node_for_camera(self, camera_id: str) -> Node:
        """找到 camera 所属节点；跨节点重复时报 409。"""
        found: list[Node] = []
        for node in self.nodes.values():
            try:
                node.store.get_camera_raw(camera_id)
            except ApiError as exc:
                if exc.status != 404:
                    raise
            else:
                found.append(node)
        if not found:
            raise ApiError(404, "NOT_FOUND", f"camera {camera_id} not found")
        if len(found) > 1:
            names = ", ".join(node.name for node in found)
            raise ApiError(
                409, "DUPLICATE_CAMERA",
                f"摄像头 {camera_id} 同时存在于节点 {names}",
            )
        return found[0]

    def _enabled_count(self, node: Node) -> int:
        return len([
            cam for cam in node.store.list_cameras({})
            if cam.get("enabled", True)
        ])

    def route_create(self, payload: dict) -> Node:
        """新建摄像头选节点：显式 node 优先，否则放当前路数较少的一侧。"""
        camera_id = payload.get("id")
        if isinstance(camera_id, str) and camera_id:
            for node in self.nodes.values():
                try:
                    node.store.get_camera_raw(camera_id)
                except ApiError as exc:
                    if exc.status != 404:
                        raise
                else:
                    raise ApiError(
                        409, "CAMERA_EXISTS", f"camera {camera_id} already exists",
                    )

        will_enable = True if payload.get("enabled") is None else bool(payload.get("enabled"))
        requested = payload.get("node")
        if requested is not None:
            node = self.nodes.get(str(requested))
            if node is None:
                raise ApiError(400, "INVALID_ARGUMENT", "node 必须是 a 或 b")
            if will_enable and self._enabled_count(node) >= self.max_cameras_per_node:
                raise ApiError(
                    409, "CAPACITY_EXCEEDED",
                    f"节点 {node.name} 已达到 {self.max_cameras_per_node} 路上限",
                )
            return node

        candidates = sorted(
            self.nodes.values(),
            key=lambda node: (self._enabled_count(node), 0 if node.name == "a" else 1),
        )
        if not will_enable:
            return candidates[0]
        for node in candidates:
            if self._enabled_count(node) < self.max_cameras_per_node:
                return node
        raise ApiError(
            409, "CAPACITY_EXCEEDED",
            f"两个节点均达到 {self.max_cameras_per_node} 路上限",
        )

    def ensure_enable_capacity(self, node: Node, camera_id: str) -> None:
        """PATCH enabled:true 时也要守住每节点上限。"""
        cameras = node.store.list_cameras({})
        current = next(
            (cam for cam in cameras if str(cam.get("id")) == str(camera_id)),
            None,
        )
        if current is None:
            raise ApiError(404, "NOT_FOUND", f"camera {camera_id} not found")
        if current.get("enabled", True):
            return
        enabled = len([cam for cam in cameras if cam.get("enabled", True)])
        if enabled >= self.max_cameras_per_node:
            raise ApiError(
                409, "CAPACITY_EXCEEDED",
                f"节点 {node.name} 已达到 {self.max_cameras_per_node} 路上限",
            )

    def list_algorithms(self) -> list[dict]:
        names = list(self.nodes)
        base_name = names[0]
        base = self.nodes[base_name].store.list_algorithms()
        for name in names[1:]:
            other = self.nodes[name].store.list_algorithms()
            if other != base:
                raise ApiError(
                    500, "CONFIG_MISMATCH",
                    f"节点 {name} 与 {base_name} 的算法目录不一致",
                )
        return base

    def health(self) -> dict[str, dict]:
        result = {}
        for name, node in self.nodes.items():
            status = _read_json(node.status_path)
            result[name] = {
                "pipeline": status.get("pipeline", "unknown"),
                "config_revision": node.store.revision(),
            }
        return result
