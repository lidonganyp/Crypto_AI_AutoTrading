import os
import tempfile
import unittest
import shutil
from pathlib import Path

from config import get_settings
from core.storage import Storage


def make_candles(count: int, base: float):
    candles = []
    for i in range(count):
        price = base + i * 0.5
        candles.append(
            {
                "timestamp": 1700000000000 + i * 3600000,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.2,
                "volume": 1000 + i * 10,
            }
        )
    return candles


class V2ArchitectureTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="cryptoai_test_")
        path = str(Path(self.temp_dir) / "test.db")
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        os.close(fd)
        self.db_path = path
        self.storage = Storage(path)
        self.settings = get_settings()

    def tearDown(self):
        if getattr(self, "temp_dir", "") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
