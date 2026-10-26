"""
2628 影子实例端到端仿真：用**真实** deploy/config.2628-test.yaml 驱动
「假帧 → 探针 → 占用状态机 → webhook payload」全链路，钉住试运行的预期行为。

不需要 GPU / DeepStream：探针与告警层都是纯 Python 判定，被替换的只有
DeepStream 元数据（鸭子类型假对象）和时钟（虚拟推进，仿真 10fps）。

这里断言的现象 = 测试机上应当观察到的现象；对不上即为缺陷。
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

import server.alert.manager as manager_mod
from server.alert.rules import parse_rules
from server.model_spec import parse_gies
from server.pipeline.probe import SafetyProbe
from server.alert.manager import AlertManager

from tests.test_zones import MUX_H, MUX_W, _Batch, _Frame, _Obj

SHADOW_CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "config.2628-test.yaml"
FPS = 10


class _FakeClock:
    """虚拟时钟：_PresenceState 依赖 time.monotonic，按帧推进而非真实等待。"""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


class _RecordingWebhook:
    def __init__(self):
        self.payloads = []

    def send(self, payload):
        self.payloads.append(payload)


def _load_shadow():
    with SHADOW_CONFIG.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return (cfg, parse_gies(cfg["model"]["gies"]), parse_rules(cfg["rules"]),
            cfg["cameras"][0])


class Test2628ShadowWalk(unittest.TestCase):
    """一个人在 2628 画面里：区域外停留 → 走进区域长留 → 走出 → 再进。"""

    # 站位取自真实区域 r9-1：覆盖画面左下；右侧为区域外
    INSIDE = (460, 800, 80, 100)      # feet=(500,900) ∈ r9-1
    OUTSIDE = (1660, 800, 80, 100)    # feet=(1700,900) ∉ r9-1

    def setUp(self):
        self._real_time = manager_mod.time
        self.clock = _FakeClock()
        manager_mod.time = self.clock

        cfg, gies, rules, cam = _load_shadow()
        self.active = list(cam["active_rules"])
        self.webhook = _RecordingWebhook()
        mgr = AlertManager("2628", cam.get("name", "2628"),
                           [rules[n] for n in self.active], webhook=self.webhook)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t")
        self.probe = SafetyProbe(
            {0: mgr}, executor=self.executor,
            gies=gies, rules=rules, person_uid=1,
            person_conf_threshold=cfg["model"]["person_conf_threshold"],
            active_rules_by_source={0: set(self.active)},
            mux_width=MUX_W, mux_height=MUX_H,
        )

    def tearDown(self):
        manager_mod.time = self._real_time
        self.executor.shutdown(wait=True)

    def _step(self, seconds, box, extra_objs=()):
        """推进 seconds 秒，每帧投递一个位于 box 的人员（可附二级检测框）。"""
        for _ in range(int(seconds * FPS)):
            objs = [_Obj(1, *box, conf=0.9), *extra_objs]
            self.probe.handle_metadata(_Batch([_Frame(0, objs)]))
            self.clock.now += 1.0 / FPS

    def _alerts(self, alert_type, expect=1, budget=5.0):
        """等 fire-and-forget 推送落地。expect=0 用短预算（等不到才符合预期）。"""
        deadline = time.time() + (budget if expect else 0.5)
        while time.time() < deadline:
            got = [p for p in self.webhook.payloads if p["alert_type"] == alert_type]
            if len(got) >= expect:
                return got
            time.sleep(0.02)
        return [p for p in self.webhook.payloads if p["alert_type"] == alert_type]

    def test_lingering_in_zone_reports_once(self):
        """走进区域后待 60 秒（600 帧）→ 只应成单 1 次。"""
        self._step(2, self.OUTSIDE)
        self._step(60, self.INSIDE)
        got = self._alerts("intrusion", expect=1)
        self.assertEqual(len(got), 1, "持续在场不应重复成单（电平方案这里会出 ~120 条）")
        self.assertEqual(got[0]["zones"], ["r9-1"], "payload 需回指标定 RuleId")
        self.assertEqual(got[0]["camera_id"], "2628")
        self.assertIsNone(got[0]["objects"][0]["attributes"][0]["bbox"])

    def test_leave_then_reenter_is_a_new_event(self):
        """走出区域（越过离开去抖）再进入 → 应为第 2 条独立事件。"""
        self._step(2, self.OUTSIDE)
        self._step(10, self.INSIDE)
        self._step(20, self.OUTSIDE)
        self._step(10, self.INSIDE)
        got = self._alerts("intrusion", expect=2)
        self.assertEqual(len(got), 2, "两次进入应为两条独立事件")

    def test_entry_debounce_blocks_brief_blip(self):
        """min_detection_count=5 → 进 4 帧就走不应成单；真进去才成单。"""
        self._step(0.4, self.INSIDE)      # 4 帧 < 5
        self._step(3, self.OUTSIDE)
        self.assertEqual(self._alerts("intrusion", expect=0), [])
        self._step(1, self.INSIDE)
        self.assertEqual(len(self._alerts("intrusion", expect=1)), 1)

    def test_no_helmet_is_frame_wide_on_2628(self):
        """2628 的安全帽规则未绑定区域：人在**区域外**不戴帽也必须成单。"""
        head = _Obj(3, 1680, 810, 30, 30, conf=0.9, label="head")
        self._step(2, self.OUTSIDE, extra_objs=[head])
        self.assertEqual(self._alerts("intrusion", expect=0), [],
                         "人在区域外不应产生入侵告警")
        self.assertTrue(self._alerts("no_helmet", expect=1),
                        "no_helmet 无 zones → 全图判定，区域外照常告警")

    def test_shadow_config_binds_intrusion_with_zone(self):
        """配置形状自检：2628 生效规则 = 全图安全帽 + 带区域的入侵。"""
        _, _, rules, cam = _load_shadow()
        active = set(cam["active_rules"])
        self.assertEqual(active, {"no_helmet", "intrusion"})
        self.assertEqual(rules["no_helmet"].zones, (), "安全帽必须保持全图")
        self.assertEqual([z.id for z in rules["intrusion"].zones], ["r9-1"])
        self.assertEqual(rules["intrusion"].alert_type, "intrusion",
                         "对外类型不得泄漏内部规则名")


if __name__ == "__main__":
    unittest.main()
