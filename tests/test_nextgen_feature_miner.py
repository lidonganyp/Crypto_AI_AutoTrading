import unittest

from nextgen_evolution.feature_miner import FeatureMiner


def make_intraday_candles(count: int, base: float = 100.0):
    candles = []
    for idx in range(count):
        drift = idx * 0.08
        swing = ((idx % 10) - 5) * 0.22
        flush = -1.6 if idx % 21 == 0 and idx > 0 else 0.0
        reclaim = 1.1 if idx % 21 == 2 else 0.0
        price = base + drift + swing + flush + reclaim
        candles.append(
            {
                "timestamp": 1700000000000 + idx * 300000,
                "open": price - 0.12,
                "high": price + 0.65,
                "low": price - (0.85 if idx % 21 == 0 else 0.55),
                "close": price + (0.24 if idx % 3 else -0.10),
                "volume": 1000 + idx * 3 + (180 if idx % 21 in {0, 1, 2} else 0),
            }
        )
    return candles


class NextGenFeatureMinerTests(unittest.TestCase):
    def test_feature_miner_builds_points_and_summary(self):
        miner = FeatureMiner()

        result = miner.mine(make_intraday_candles(96))

        self.assertGreater(len(result.points), 50)
        self.assertIn("feature_points", result.summary)
        self.assertGreater(result.summary["feature_points"], 50)
        self.assertIn("breakout_ready_points", result.summary)
        self.assertIn("reclaim_ready_points", result.summary)
        self.assertTrue(any(point.range_expansion > 0 for point in result.points))


if __name__ == "__main__":
    unittest.main()
