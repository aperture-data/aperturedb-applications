"""Base class for the generators handed to aperturedb's ParallelLoader / ParallelQuery."""
import threading

from aperturedb.Subscriptable import Subscriptable


class Generator(Subscriptable):
    """Subclasses implement ``__len__`` and ``build(idx) -> (commands, blobs)``.

    The SDK peeks at item 0 several times before and during a run; building it once keeps
    downloads and statistics from being counted repeatedly.
    """

    def __init__(self):
        self._first = None
        self._first_lock = threading.Lock()

    def getitem(self, idx):
        if idx != 0:
            return self.build(idx)
        with self._first_lock:
            if self._first is None:
                self._first = self.build(0)
            return self._first

    def build(self, idx):
        raise NotImplementedError
