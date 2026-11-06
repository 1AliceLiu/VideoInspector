import unittest
import threading
import time
from server.pipeline.evidence_sync import EvidenceSync

class Frame:
    def bgr(self): return "pixels"

class TestEvidenceSync(unittest.TestCase):
    def test_exact_match(self):
        sync = EvidenceSync(); item = sync.request(1, 10, 100)
        sync.publish(1, 10, 100, Frame())
        self.assertEqual(item.bgr(), "pixels")
    def test_wrong_source(self):
        sync = EvidenceSync(); item = sync.request(1, 10, 100, .01)
        sync.publish(2, 10, 100, Frame())
        self.assertIsNone(item.bgr())
    def test_wrong_frame(self):
        sync = EvidenceSync(); item = sync.request(1, 10, 100, .01)
        sync.publish(1, 9, 100, Frame())
        self.assertIsNone(item.bgr())
    def test_wrong_timestamp(self):
        sync = EvidenceSync(); item = sync.request(1, 10, 100, .01)
        sync.publish(1, 10, 99, Frame())
        self.assertIsNone(item.bgr())
    def test_async_arrival(self):
        sync = EvidenceSync(); item = sync.request(1, 10, 100, .5)
        thread = threading.Thread(target=lambda: (time.sleep(.02), sync.publish(1,10,100,Frame())))
        thread.start(); self.assertEqual(item.bgr(), "pixels");thread.join()
    def test_timeout_late_frame(self):
        sync = EvidenceSync(); item = sync.request(1, 10, 100, .01)
        self.assertIsNone(item.bgr());sync.publish(1,10,100,Frame())
        self.assertIsNone(item.bgr());self.assertEqual(len(sync.pending),0)
    def test_limit_and_reuse(self):
        sync=EvidenceSync(limit=1);a=sync.request(1,1,100)
        self.assertIs(a,sync.request(1,1,100))
        self.assertIsNone(sync.request(1,2,200).bgr())
    def test_invalid_frame(self):
        self.assertIsNone(EvidenceSync().request(1,-1,0).bgr())

class TestEvidenceDelivery(unittest.TestCase):
    def test_timeout_preserves_alarm_without_wrong_image(self):
        from server.alert.manager import AlertManager
        from server.pipeline.evidence_sync import PendingEvidence
        sent = []; done = threading.Event()
        manager = AlertManager(camera_id="test", camera_name="test", rules=[], evidence_sync=True)
        manager._send_payload = lambda *a: (sent.append(a), done.set())
        manager._build_and_send("no_helmet", "no_helmet", PendingEvidence(0), [], "now")
        self.assertTrue(done.wait(1)); self.assertIsNone(sent[0][2])

    def test_matched_frame_encodes_jpeg(self):
        import numpy as np
        from server.alert.manager import AlertManager
        from server.pipeline.evidence_sync import PendingEvidence
        class Pixels:
            def bgr(self): return np.zeros((32,32,3), dtype=np.uint8)
        item=PendingEvidence(1); item.resolve(Pixels())
        sent=[];done=threading.Event()
        manager=AlertManager(camera_id="test",camera_name="test",rules=[],evidence_sync=True)
        manager._send_payload=lambda *a: (sent.append(a),done.set())
        manager._build_and_send("no_helmet","no_helmet",item,[],"now")
        self.assertTrue(done.wait(1));self.assertTrue(sent[0][2].startswith(bytes([255,216])))
