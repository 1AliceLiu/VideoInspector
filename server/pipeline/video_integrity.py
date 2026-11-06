"""Opt-in source protection: keyframe recovery before decoding, no RTP shedding."""

class KeyframeState:
    def __init__(self):
        self.ready = False
        self.dropped = 0

    def reset(self):
        self.ready = False
        self.dropped = 0

    def accept(self, delta=False, corrupted=False):
        if corrupted:
            self.ready = False
        if corrupted or (not self.ready and delta):
            self.dropped += 1
            return False
        self.ready = True
        return True


def protected_source_factory():
    # Lazy imports keep the legacy path and offline config tooling unchanged.
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GObject
    from loguru import logger

    factory_name = "viintegritysrc"
    Gst.init(None)
    if Gst.ElementFactory.find(factory_name):
        return factory_name
    base = type(Gst.ElementFactory.make("nvurisrcbin"))

    class IntegritySource(base):
        __gtype_name__ = "VideoInspectorIntegritySource"
        __gstmetadata__ = ("Video Integrity Source", "Source/Video",
                           "Recover on keyframes and suppress flagged corrupt output", "VideoInspector")

        def __init__(self):
            super().__init__()
            self.connect("child-added", self._on_child)

        def _on_child(self, proxy, child, name):
            try:
                if GObject.signal_lookup("child-added", child.__gtype__):
                    child.connect("child-added", self._on_child)
                factory = child.get_factory()
                if factory and factory.get_name() in ("h264parse", "h265parse"):
                    state = KeyframeState()

                    def gate(pad, info):
                        if info.type & Gst.PadProbeType.EVENT_DOWNSTREAM:
                            event = info.get_event()
                            if event and event.type in (Gst.EventType.FLUSH_STOP, Gst.EventType.SEGMENT):
                                state.reset()
                            return Gst.PadProbeReturn.OK
                        buf = info.get_buffer()
                        if buf is None:
                            return Gst.PadProbeReturn.OK
                        was_ready = state.ready
                        allowed = state.accept(
                            bool(buf.has_flags(Gst.BufferFlags.DELTA_UNIT)),
                            bool(buf.has_flags(Gst.BufferFlags.CORRUPTED)))
                        if not allowed:
                            return Gst.PadProbeReturn.DROP
                        if not was_ready:
                            logger.info("video-integrity keyframe-ready source={} parser={} dropped={}",
                                        self.get_name(), name, state.dropped)
                        return Gst.PadProbeReturn.OK

                    child.get_static_pad("src").add_probe(
                        Gst.PadProbeType.BUFFER | Gst.PadProbeType.EVENT_DOWNSTREAM |
                        Gst.PadProbeType.EVENT_FLUSH, gate)
                if child.find_property("discard-corrupted-frames"):
                    child.set_property("discard-corrupted-frames", True)
                    if child.find_property("automatic-request-sync-points"):
                        child.set_property("automatic-request-sync-points", True)
            except Exception:
                logger.exception("video-integrity source protection failed source={} child={}",
                                 self.get_name(), name)

    GObject.type_register(IntegritySource)
    if not Gst.Element.register(None, factory_name, Gst.Rank.NONE, IntegritySource):
        raise RuntimeError("Cannot register video-integrity source")
    return factory_name
