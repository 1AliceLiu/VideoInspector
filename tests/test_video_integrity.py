import unittest
from unittest.mock import Mock, patch
from server.pipeline.builder import PipelineBuilder

class FakePipeline:
    def __init__(self):self.elements={};self.links=[]
    def add(self,kind,name,props=None):self.elements[name]=(kind,props or {})
    def link(self,*args):self.links.append(args)

class TestVideoIntegrity(unittest.TestCase):
    def build(self,enabled):
        b=PipelineBuilder.__new__(PipelineBuilder)
        b.config={"source":{"preserve_bitstream":enabled,"rtsp_protocol":4,"drop_frame_interval":2}}
        b.cameras=[{"rtsp_url":"rtsp://invalid/one"},{"rtsp_url":"rtsp://invalid/two"}]
        p=FakePipeline()
        with patch("server.pipeline.builder._supported_nvurisrcbin_props",return_value=None), patch("server.pipeline.video_integrity.protected_source_factory",return_value="viintegritysrc"):
            b._add_sources(p,Mock())
        return p
    def test_default_path_keeps_existing_behavior(self):
        p=self.build(False)
        self.assertNotIn("drop-on-latency",p.elements["src0"][1])
        self.assertNotIn("decoded-q0",p.elements)
        self.assertIn((("src0","mux"),("","sink_%u")),p.links)
    def test_preserves_compressed_packets(self):
        p=self.build(True)
        for i in range(2):
            props=p.elements[f"src{i}"][1]
            self.assertIs(props["drop-on-latency"],False)
            self.assertEqual(props["select-rtp-protocol"],4)
            self.assertEqual(props["drop-frame-interval"],2)
    def test_bounded_queue_is_after_decoder(self):
        p=self.build(True)
        for i in range(2):
            kind,props=p.elements[f"decoded-q{i}"]
            self.assertEqual(kind,"queue")
            self.assertEqual(props,{"max-size-buffers":2,"max-size-bytes":0,"max-size-time":0,"leaky":2})
            self.assertIn((f"src{i}",f"decoded-q{i}"),p.links)
            self.assertIn(((f"decoded-q{i}","mux"),("","sink_%u")),p.links)

class TestKeyframeRecovery(unittest.TestCase):
    def state(self):
        from server.pipeline.video_integrity import KeyframeState
        return KeyframeState()
    def test_delta_frames_wait_for_keyframe(self):
        state=self.state()
        self.assertFalse(state.accept(delta=True));self.assertFalse(state.accept(delta=True))
        self.assertEqual(state.dropped,2)
        self.assertTrue(state.accept());self.assertTrue(state.accept(delta=True))
    def test_corruption_requires_new_keyframe(self):
        state=self.state();self.assertTrue(state.accept())
        self.assertFalse(state.accept(corrupted=True))
        self.assertFalse(state.accept(delta=True));self.assertTrue(state.accept())
    def test_corrupted_keyframe_does_not_unlock(self):
        state=self.state();self.assertFalse(state.accept(corrupted=True))
        self.assertFalse(state.accept(delta=True))
    def test_reconnect_reset(self):
        state=self.state();state.accept();state.reset()
        self.assertFalse(state.accept(delta=True))
