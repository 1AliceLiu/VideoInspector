"""Bounded, demand-only rendezvous for decision metadata and downstream pixels."""
import threading
import time


class PendingEvidence:
    def __init__(self, timeout):
        self.deadline = time.monotonic() + timeout
        self.event = threading.Event()
        self.frame = None
        self.lock = threading.Lock()

    def resolve(self, frame):
        self.frame = frame
        self.event.set()

    def bgr(self):
        self.event.wait(max(0, self.deadline - time.monotonic()))
        with self.lock:
            return self.frame.bgr() if self.frame is not None else None


class EvidenceSync:
    def __init__(self, limit=16):
        self.pending = {}
        self.lock = threading.Lock()
        self.limit = limit

    def request(self, source, number, pts, timeout=2.0):
        key = (source, number, pts)
        with self.lock:
            self._expire()
            if key in self.pending:
                return self.pending[key]
            item = PendingEvidence(timeout)
            if number < 0 or sum(k[0] == source for k in self.pending) >= self.limit:
                item.resolve(None)
                return item
            self.pending[key] = item
            return item

    def publish(self, source, number, pts, frame):
        with self.lock:
            self._expire()
            item = self.pending.pop((source, number, pts), None)
            if item is not None:
                item.resolve(frame)

    def _expire(self):
        for key, item in list(self.pending.items()):
            if time.monotonic() >= item.deadline:
                self.pending.pop(key)
                item.resolve(None)
