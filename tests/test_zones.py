"""
区域（zone）功能单测：几何判定 / 探针区域门 / presence 占用状态机 / 配置校验

不依赖 DeepStream 运行期：探针用鸭子类型的假 frame_meta 驱动 handle_metadata，
只验证我们自己的判定逻辑与 OSD 决策。
"""

import time
import unittest

from server.alert.regions import (
    Zone, ZoneMatcher, anchor_point, parse_zones, point_in_polygon,
)
from server.alert.rules import parse_rules, RuleConfig, TRIGGER_PRESENCE
from server.alert.manager import AlertManager, _PresenceState, _RuleState
from server.metadata import AttributeMeta, ObjectMeta
from server.model_spec import parse_gies
from server.config import validate_config
from server.pipeline.probe import SafetyProbe

from pathlib import Path

# 2628-area.txt 的 r9-1（左下整块地面区域，归一化坐标）
AREA_2628 = [
    [0.2943, 0.3311], [0.4398, 0.3408], [0.5252, 0.6883],
    [0.5066, 0.9835], [0.0055, 0.9874], [0.0044, 0.6981],
]
MUX_W, MUX_H = 1920, 1080


# --------------------------------------------------------------------- 假对象
class _Rect:
    def __init__(self, left, top, width, height):
        self.left, self.top = left, top
        self.width, self.height = width, height
        self.border_width = 3
        self.border_color = None


class _Text:
    def __init__(self):
        self.display_text = "default"
        self.x_offset = 0
        self.y_offset = 0
        class _Font:
            name = None
            size = 0
            color = None
        self.font_params = _Font()


class _ClfLabel:
    """模拟 NvDsClassifierMeta 的一项（get_n_label 返回 'label conf' 串）。"""
    def __init__(self, uid, raw):
        self.unique_component_id = uid
        self._raw = raw
        self.n_labels = 1

    def get_n_label(self, i):
        return self._raw


class _Obj:
    def __init__(self, uid, left, top, w, h, conf=0.9, label="person", clf=()):
        self.unique_component_id = uid
        self.rect_params = _Rect(left, top, w, h)
        self.text_params = _Text()
        self.confidence = conf
        self.label = label
        self.classifier_items = list(clf)


class _Frame:
    def __init__(self, source_id, objs):
        self.source_id = source_id
        self.frame_number = 1
        self.buffer_pts = 1
        self._objs = objs

    @property
    def object_items(self):
        # 一次性迭代器语义（与真实探针一致：不能跨 pass 重复取）
        return iter(self._objs)


class _Batch:
    def __init__(self, frames):
        self._frames = frames

    @property
    def frame_items(self):
        return iter(self._frames)


class _Manager:
    """记录探针投递的 ObjectMeta，并暴露 camera_name。"""
    def __init__(self, camera_name="cam"):
        self.camera_name = camera_name
        self.received = []

    def handle(self, objects, snapshot=None, executor=None, frame_info=None):
        self.received.append(objects)
        return False


def _gies_and_rules(rules_raw, gies_raw=None):
    gies_raw = gies_raw or {
        "person": {"kind": "detector", "source": "models/person/x.pt", "uid": 1},
        "helmet": {"kind": "detector", "source": "models/helmet/x.pt", "uid": 3,
                   "violation": "head"},
        "vest_cls": {"kind": "classifier", "source": "models/vest/x.pt", "uid": 6,
                     "violation": "no_vest"},
    }
    return parse_gies(gies_raw), parse_rules(rules_raw)


def _zone_rules(with_vest_zone=True):
    """影子实例同形的规则集：intrusion(带区) + no_helmet(全图) + no_vest(带区)。"""
    rules = {
        "intrusion": {
            "gie": "person", "trigger": "presence", "cooldown_seconds": 0,
            "min_detection_count": 5, "vacant_frames": 50,
            "attribute_threshold": 0.5, "alert_type": "intrusion",
            "zones": [{"id": "r9-1", "anchor": "feet", "points": AREA_2628}],
        },
        "no_helmet": {"gie": "helmet", "cooldown_seconds": 10,
                      "min_detection_count": 3, "attribute_threshold": 0.5},
    }
    if with_vest_zone:
        rules["no_vest_zone"] = {
            "gie": "vest_cls", "cooldown_seconds": 60, "min_detection_count": 10,
            "attribute_threshold": 0.8, "alert_type": "no_vest",
            "zones": [{"id": "r9-1", "anchor": "feet", "points": AREA_2628}],
        }
    return rules


# ===================================================================== 几何
class TestGeometry(unittest.TestCase):

    def test_real_area_parses(self):
        zs = parse_zones([{"id": "r9-1", "points": AREA_2628}], where="t")
        self.assertEqual(len(zs), 1)
        # 6 个顶点必须原样保留（禁止共线简化：贴边顶点表达「区域伸出画面」）
        self.assertEqual(len(zs[0].points), 6)
        self.assertEqual(zs[0].anchor, "feet")

    def test_platform_dict_points_accepted(self):
        pts = [{"X": p[0], "Y": p[1]} for p in AREA_2628]
        zs = parse_zones([{"RuleId": "r9-1", "Points": pts}], where="t")
        self.assertEqual(zs[0].id, "r9-1")

    def test_rejects_bad_zones(self):
        for bad, why in (
            ([{"points": [[0, 0], [1, 1]]}], "少于 3 点"),
            ([{"points": [[0, 0], [1, 1], [2, 2]]}], "坐标越界"),
            ([{"points": [[0, 0], [1, 1], [0, 0], [0, 1], [1, 0]]}], "边自交"),
            ([{"id": "same", "points": [[0, 0], [1, 1], [0, 1]]},
              {"id": "same", "points": [[0, 0], [1, 0], [1, 1]]}], "id 重复"),
        ):
            with self.assertRaises(ValueError, msg=why):
                parse_zones(bad, where="t")

    def test_dedupes_closed_ring_repeat(self):
        ring = AREA_2628 + [AREA_2628[0]]      # 首尾重复（闭合点）
        zs = parse_zones([{"points": ring}], where="t")
        self.assertEqual(len(zs[0].points), 6)

    def test_point_in_polygon(self):
        sq = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        self.assertTrue(point_in_polygon(0.5, 0.5, sq))
        self.assertFalse(point_in_polygon(1.5, 0.5, sq))

    def test_anchor_point(self):
        self.assertEqual(anchor_point(10, 20, 30, 80, "feet"), (20.0, 80.0))
        self.assertEqual(anchor_point(10, 20, 30, 80, "center"), (20.0, 50.0))

    def test_normalized_maps_linearly_to_mux(self):
        """归一化 → 像素：区域中心点换算后仍在区域内（拉伸坐标系的推导基础）。"""
        m = ZoneMatcher(parse_zones([{"points": AREA_2628}], where="t"), MUX_W, MUX_H)
        inside = m.hit((0.25 * MUX_W, 0.8 * MUX_H), (0.25 * MUX_W, 0.8 * MUX_H))
        self.assertIsNotNone(inside)
        outside = m.hit((0.9 * MUX_W, 0.1 * MUX_H), (0.9 * MUX_W, 0.1 * MUX_H))
        self.assertIsNone(outside)

    def test_bbox_quickreject_matches_plain_pip(self):
        """外接矩形粗筛不能改变结果（与直接射线法逐点等价）。"""
        zones = parse_zones([{"points": AREA_2628}], where="t")
        m = ZoneMatcher(zones, MUX_W, MUX_H)
        for gx in range(0, 21):
            for gy in range(0, 21):
                p = (gx / 20 * MUX_W, gy / 20 * MUX_H)
                direct = point_in_polygon(p[0], p[1], tuple(
                    (x * MUX_W, y * MUX_H) for x, y in zones[0].points))
                self.assertEqual(m.hit(p, p) is not None, direct)


# ============================================================= 规则/配置校验
class TestRuleValidation(unittest.TestCase):

    def test_presence_template_without_zones_needs_camera_binding(self):
        """基础规则可作无 zones 的模板声明，但**任何启用它的摄像头必须补上区域**。"""
        base = {"gie": "person", "trigger": "presence", "cooldown_seconds": 0,
                "min_detection_count": 5, "attribute_threshold": 0.5}
        # 模板本身可解析（parse_rules 不管摄像头维度）
        r = parse_rules({"intrusion": base})
        self.assertEqual(r["intrusion"].zones, ())

        cfg = {
            "model": {"gies": {
                "person": {"kind": "detector", "source": "models/person/a.pt", "uid": 1},
            }},
            "rules": {"intrusion": base},
            "cameras": [{"id": "2628", "rtsp_url": "rtsp://a", "enabled": True,
                         "active_rules": ["intrusion"]}],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg, Path("<test>"))
        self.assertIn("未绑定 zones", str(ctx.exception))

        # 同一模板 + 按摄像头生成的带区域实例 → 合法
        gen = dict(base, zones=[{"id": "r9-1", "points": AREA_2628}])
        cfg["rules"]["__cam__2628__ab12__intrusion"] = gen
        cfg["cameras"][0]["active_rules"] = ["__cam__2628__ab12__intrusion"]
        validate_config(cfg, Path("<test>"))

    def test_violation_without_zones_is_fullframe(self):
        r = parse_rules({"no_vest": {"gie": "vest_cls", "cooldown_seconds": 1,
                                     "min_detection_count": 1,
                                     "attribute_threshold": 0.5}})
        self.assertEqual(r["no_vest"].zones, ())
        self.assertEqual(r["no_vest"].trigger, "violation")

    def test_unknown_trigger_rejected(self):
        with self.assertRaises(ValueError):
            parse_rules({"x": {"gie": "g", "trigger": "magic", "cooldown_seconds": 1,
                               "min_detection_count": 1, "attribute_threshold": 0.5}})

    def test_presence_must_bind_anchor_gie(self):
        cfg = {
            "model": {"gies": {
                "person": {"kind": "detector", "source": "models/person/a.pt", "uid": 1},
                "helmet": {"kind": "detector", "source": "models/helmet/a.pt",
                           "uid": 3, "violation": "head"},
            }},
            "rules": {"intrusion": {
                "gie": "helmet", "trigger": "presence", "cooldown_seconds": 0,
                "min_detection_count": 5, "attribute_threshold": 0.5,
                "zones": [{"points": AREA_2628}]}},
            "cameras": [{"id": "1", "rtsp_url": "rtsp://a", "enabled": True}],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg, Path("<test>"))
        self.assertIn("锚点检测器", str(ctx.exception))

    def test_2628_config_is_valid(self):
        import yaml
        with open("deploy/config.yaml", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        validate_config(raw, Path("deploy/config.yaml"))


# ==================================================================== 探针
class TestProbeZoneGate(unittest.TestCase):

    def _probe(self, objs_by_src, active_extra=()):
        gies, rules = _gies_and_rules(_zone_rules())
        mgr = {0: _Manager("2628")}
        active = {0: {"intrusion", "no_helmet", "no_vest_zone", *active_extra}}
        probe = SafetyProbe(mgr, gies=gies, rules=rules, person_uid=1,
                            person_conf_threshold=0.6,
                            active_rules_by_source=active,
                            mux_width=MUX_W, mux_height=MUX_H)
        probe.handle_metadata(_Batch([_Frame(0, objs_by_src)]))
        return mgr[0].received[0]

    def test_person_inside_zone_gets_intrusion_attr(self):
        # 脚点 (500, 900) 落在 r9-1 内（区域覆盖画面左下）
        p = _Obj(1, 460, 800, 80, 100, conf=0.9)
        objs = self._probe([p])
        self.assertEqual(len(objs), 1)
        names = {a.name for a in objs[0].attributes}
        self.assertIn("intrusion", names)
        zones = [a.zone for a in objs[0].attributes if a.name == "intrusion"]
        self.assertEqual(zones, ["r9-1"])

    def test_person_outside_zone_gets_no_attr(self):
        p = _Obj(1, 1660, 800, 80, 100, conf=0.9)     # 画面右侧，区域外
        objs = self._probe([p])
        # 探针照常产出 person 实体（供其它全图规则使用），但本例无任何规则命中
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0].attributes, ())

    def test_low_conf_person_ignored(self):
        p = _Obj(1, 460, 800, 80, 100, conf=0.3)
        self.assertEqual(self._probe([p]), [])

    def test_violation_rule_filtered_by_zone(self):
        """no_vest 违规 + 人在区域外 → 不报；在区域内 → 报。同模型输出，只差位置。"""
        inside = _Obj(1, 460, 800, 80, 100,
                      clf=(_ClfLabel(6, "no_vest 0.9"),))
        attrs = {a.name for a in self._probe([inside])[0].attributes}
        self.assertIn("no_vest_zone", attrs)

        outside = _Obj(1, 1660, 800, 80, 100,
                       clf=(_ClfLabel(6, "no_vest 0.9"),))
        got = self._probe([outside])[0]
        # 分类器同样判「未穿反光衣」，但人在区域外 → 该规则不命中
        self.assertNotIn("no_vest_zone", {a.name for a in got.attributes})
        self.assertNotIn("intrusion", {a.name for a in got.attributes})

    def test_fullframe_rule_unaffected_by_zone(self):
        """no_helmet（无 zones）在区域外照常报 —— 全图规则不受区域功能影响。"""
        head = _Obj(3, 1680, 810, 30, 30, conf=0.9, label="head")
        person = _Obj(1, 1660, 800, 80, 100, conf=0.9)
        objs = self._probe([person, head])
        attrs = {a.name for a in objs[0].attributes}
        self.assertIn("no_helmet", attrs)
        self.assertNotIn("intrusion", attrs)

    def test_absent_zone_feature_is_noop(self):
        """回归红线：完全没有 zones/presence 时，行为与原实现一致。"""
        gies, rules = _gies_and_rules({
            "no_helmet": {"gie": "helmet", "cooldown_seconds": 10,
                          "min_detection_count": 3, "attribute_threshold": 0.5},
            "no_vest": {"gie": "vest_cls", "cooldown_seconds": 300,
                        "min_detection_count": 15, "attribute_threshold": 0.8},
        })
        mgr = {0: _Manager()}
        probe = SafetyProbe(mgr, gies=gies, rules=rules, person_uid=1,
                            person_conf_threshold=0.6,
                            active_rules_by_source={0: {"no_helmet", "no_vest"}},
                            mux_width=MUX_W, mux_height=MUX_H)
        head = _Obj(3, 20, 20, 30, 30, conf=0.9, label="head")
        person = _Obj(1, 0, 10, 80, 100, conf=0.9,
                      clf=(_ClfLabel(6, "no_vest 0.9"),))
        probe.handle_metadata(_Batch([_Frame(0, [person, head])]))
        got = mgr[0].received[0]
        self.assertEqual({a.name for a in got[0].attributes}, {"no_helmet", "no_vest"})
        self.assertIsNone(got[0].attributes[0].zone)

    def test_presence_rule_does_not_pollute_detector_buckets(self):
        """person uid 不该被当成整帧检测器收集（presence 走独立分支）。"""
        gies, rules = _gies_and_rules(_zone_rules())
        probe = SafetyProbe({}, gies=gies, rules=rules, person_uid=1,
                            active_rules_by_source={}, mux_width=MUX_W,
                            mux_height=MUX_H)
        probe.handle_metadata(_Batch([_Frame(0, [_Obj(1, 460, 800, 80, 100)])]))
        self.assertIn("intrusion", probe._presence_rules)
        self.assertNotIn("no_helmet", probe._presence_rules)


# ============================================================== 占用状态机
class TestPresenceStateMachine(unittest.TestCase):
    """时间线语义（无 tracker 的前提下，一次入侵 = 一条告警）。"""

    P = ObjectMeta("person", 0.9, (10, 10, 50, 90))

    def _rule(self, min_c=5, vacant=50, cool=0.0):
        return RuleConfig("i", "person", cool, min_c, 0.5,
                          trigger=TRIGGER_PRESENCE, vacant_frames=vacant)

    def _run(self, windows, seconds, fps=10, **kw):
        st = _PresenceState(self._rule(**kw))
        fired = []
        for frame in range(int(seconds * fps)):
            t = frame / fps
            inside = any(a <= t < b for a, b in windows)
            if st.handle([self.P] if inside else []):
                fired.append(round(t, 2))
        return fired

    def test_linger_once(self):
        """待 30 分钟也只报 1 条（电平方案的对照实验见场景 B）。"""
        self.assertEqual(self._run([(0, 1800)], 1800), [0.4])

    def test_two_separate_events_both_report(self):
        """甲进→走→乙再进：两次独立事件各报一次，**不得被冷却吞掉**。"""
        self.assertEqual(self._run([(0, 10), (20, 30)], 60), [0.4, 20.4])

    def test_entry_debounce_blocks_flicker(self):
        self.assertEqual(self._run([(5, 5.2)], 20), [])          # 2 帧误检

    def test_vacant_debounce_blocks_detection_drop(self):
        """区域内人员被遮挡 4 帧（< vacant_frames）不算离开，不重复报。"""
        self.assertEqual(self._run([(0, 5), (5.4, 30)], 30), [0.4])

    def test_group_entry_reports_once(self):
        """一波 5 人进入 → 1 条（状态机按规则而非按人）。"""
        st = _PresenceState(self._rule())
        fired = 0
        for _ in range(60):
            if st.handle([self.P] * 5):
                fired += 1
        self.assertEqual(fired, 1)


class TestViolationStateMachineUnchanged(unittest.TestCase):
    """现有 PPE 语义不得被本次改动影响（冷却行为保持原样）。"""

    P = ObjectMeta("person", 0.9, (10, 10, 50, 90))

    def test_repeats_after_cooldown(self):
        rule = RuleConfig("no_vest", "vest_cls", 30.0, 5, 0.5)
        st = _RuleState(rule)
        fired = 0
        for _ in range(10):
            if st.handle([self.P]):
                fired += 1
        self.assertEqual(fired, 1)


# ================================================================= payload
class TestPayloadZone(unittest.TestCase):

    def test_payload_carries_zone_and_evidence_has_zones(self):
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        zone = Zone(id="r9-1", points=tuple((float(x), float(y)) for x, y in AREA_2628))
        rule = RuleConfig("i", "person", 0.0, 1, 0.5, trigger=TRIGGER_PRESENCE,
                          zones=(zone,), alert_type="intrusion")
        sent = {}

        class FakeWebhook:
            def send(self, payload):
                sent.update(payload)

        mgr = AlertManager("2628", "AI Camera2628", [rule], webhook=FakeWebhook())
        objs = [ObjectMeta("person", 0.9, (460, 800, 540, 900),
                           attributes=(AttributeMeta("i", 0.9, zone="r9-1"),))]
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 推送是 executor + daemon 线程的 fire-and-forget，测试需等它落地
        ex = ThreadPoolExecutor(max_workers=1)
        self.assertTrue(mgr.handle(objs, snapshot=frame, executor=ex))
        ex.shutdown(wait=True)
        for _ in range(100):
            if sent:
                break
            time.sleep(0.02)

        self.assertEqual(sent["alert_type"], "intrusion")
        self.assertEqual(sent["zones"], ["r9-1"])
        self.assertEqual(sent["objects"][0]["attributes"][0]["zone"], "r9-1")
        self.assertIsNotNone(sent["frame_base64"])

    def test_evidence_render_scales_normalized_zone(self):
        """证据图区域轮廓按**帧实际宽高**缩放：换分辨率不影响绘制。"""
        import numpy as np
        zone = Zone(id="r9-1", points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        for shape in ((1080, 1920, 3), (720, 1280, 3)):
            frame = np.zeros(shape, dtype=np.uint8)
            out = AlertManager._render_evidence(frame, "intrusion", [], (zone,))
            self.assertEqual(out.shape, shape)
            self.assertGreater(int((out > 0).sum()), 0)   # 区域线确实被画出来了


if __name__ == "__main__":
    unittest.main()
