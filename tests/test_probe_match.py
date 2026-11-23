"""head/helmet 双框合规覆盖单测（server/pipeline/probe._match_violation）。"""
import unittest
from types import SimpleNamespace

from server.pipeline.probe import (SafetyProbe, _COMPLIANT_COVER,
                                   _COMPLIANT_OVERRIDE_CONF)


def person(left, top, width, height):
    return SimpleNamespace(rect_params=SimpleNamespace(
        left=left, top=top, width=width, height=height))


def box(label, conf, rect):
    x1, y1, x2, y2 = rect
    return (label, conf, (x1 + x2) / 2, (y1 + y2) / 2, rect)


class TestMatchViolation(unittest.TestCase):
    person = person(100, 100, 100, 300)          # x:100-200, y:100-400

    def match(self, boxes, label="head", thr=0.5):
        return SafetyProbe._match_violation(self.person, boxes, label, thr)

    def test_head_alone(self):
        head = box("head", 0.60, (130, 110, 160, 150))
        self.assertEqual(self.match([head]), 0.60)

    def test_head_covered_by_confident_helmet_suppressed(self):
        head = box("head", 0.60, (130, 110, 160, 150))
        helmet = box("helmet", 0.55, (128, 108, 162, 152))
        self.assertIsNone(self.match([head, helmet]))

    def test_helmet_below_override_conf_does_not_suppress(self):
        head = box("head", 0.60, (130, 110, 160, 150))
        helmet = box("helmet", _COMPLIANT_OVERRIDE_CONF - 0.01, (128, 108, 162, 152))
        self.assertEqual(self.match([head, helmet]), 0.60)

    def test_helmet_not_overlapping_does_not_suppress(self):
        head = box("head", 0.60, (130, 110, 160, 150))
        helmet = box("helmet", 0.90, (170, 300, 195, 340))
        self.assertEqual(self.match([head, helmet]), 0.60)

    def test_large_helmet_covering_head_suppresses(self):
        head = box("head", 0.60, (130, 110, 160, 150))
        helmet = box("helmet", 0.80, (110, 95, 200, 160))   # 覆盖 head 面积 > 50%
        self.assertIsNone(self.match([head, helmet]))

    def test_partial_cover_below_threshold_keeps_head(self):
        head = box("head", 0.60, (130, 110, 160, 150))     # 30x40
        # 只覆盖 head 的 1/3 面积（横向 10/30）
        helmet = box("helmet", 0.90, (145, 110, 155, 150))
        self.assertEqual(self.match([head, helmet]), 0.60)

    def test_second_head_uncovered_still_fires(self):
        head_a = box("head", 0.70, (130, 110, 160, 150))
        helmet = box("helmet", 0.60, (128, 108, 162, 152))
        head_b = box("head", 0.52, (135, 250, 165, 290))
        self.assertEqual(self.match([head_a, helmet, head_b]), 0.52)

    def test_head_below_attribute_threshold(self):
        head = box("head", 0.49, (130, 110, 160, 150))
        self.assertIsNone(self.match([head]))

    def test_head_outside_person(self):
        head = box("head", 0.90, (300, 110, 330, 150))
        self.assertIsNone(self.match([head]))

    def test_cover_ratio_math(self):
        from server.pipeline.probe import _cover_ratio
        self.assertAlmostEqual(_cover_ratio((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(_cover_ratio((0, 0, 10, 10), (0, 0, 5, 10)), 0.5)
        self.assertEqual(_cover_ratio((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(_cover_ratio((0, 0, 0, 0), (0, 0, 10, 10)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
