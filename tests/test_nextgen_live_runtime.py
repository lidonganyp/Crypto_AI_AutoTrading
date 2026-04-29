import tempfile
import unittest
from pathlib import Path

from pydantic import SecretStr

from config import Settings
from core.storage import Storage
from nextgen_evolution import AutonomyLiveRuntime, EvolutionConfig


class FakeAdapter:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class NextGenLiveRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "cryptoai.db"
        self.storage = Storage(str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_config_normalizes_whitelist_from_settings(self):
        settings = Settings()
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT:USDT"]
        settings.exchange.max_active_symbols = 3
        settings.risk.max_positions = 2
        runtime = AutonomyLiveRuntime(self.storage, settings=settings)

        config = runtime.build_config(
            base_config=EvolutionConfig(autonomy_live_max_active_runtimes=5),
            requested_live=True,
        )

        self.assertTrue(config.autonomy_live_enabled)
        self.assertEqual(
            config.autonomy_live_whitelist,
            ("BTC/USDT:USDT", "ETH/USDT:USDT"),
        )
        self.assertEqual(config.autonomy_live_max_active_runtimes, 2)

    def test_operator_request_persists_and_normalizes_controls(self):
        settings = Settings()
        runtime = AutonomyLiveRuntime(self.storage, settings=settings)

        persisted = runtime.set_operator_request(
            requested_live=True,
            whitelist=["BTC/USDT", "ETH/USDT:USDT"],
            max_active_runtimes=3,
            reason="cli",
        )

        self.assertTrue(persisted["requested_live"])
        self.assertEqual(
            persisted["whitelist"],
            ("BTC/USDT:USDT", "ETH/USDT:USDT"),
        )
        self.assertEqual(persisted["max_active_runtimes"], 3)
        self.assertEqual(self.storage.get_state(runtime.REQUESTED_ENABLED_STATE_KEY), "true")
        self.assertEqual(self.storage.get_state(runtime.REQUESTED_REASON_STATE_KEY), "cli")

    def test_resolve_operator_request_prefers_explicit_overrides(self):
        settings = Settings()
        runtime = AutonomyLiveRuntime(self.storage, settings=settings)
        runtime.set_operator_request(
            requested_live=True,
            whitelist=["BTC/USDT"],
            max_active_runtimes=2,
            reason="cli",
        )

        resolved = runtime.resolve_operator_request(
            requested_live=False,
            whitelist=["ETH/USDT"],
            max_active_runtimes=1,
        )

        self.assertFalse(resolved["requested_live"])
        self.assertEqual(resolved["whitelist"], ("ETH/USDT:USDT",))
        self.assertEqual(resolved["max_active_runtimes"], 1)

    def test_evaluate_blocks_live_when_operator_requirements_fail(self):
        settings = Settings()
        settings.app.runtime_mode = "paper"
        settings.app.allow_live_orders = False
        settings.exchange.symbols = ["BTC/USDT"]
        runtime = AutonomyLiveRuntime(self.storage, settings=settings)
        config = runtime.build_config(
            requested_live=True,
        )

        status = runtime.evaluate(
            requested_live=True,
            config=config,
        )

        self.assertFalse(status.effective_live)
        self.assertTrue(status.dry_run)
        self.assertFalse(status.allow_entries)
        self.assertFalse(status.allow_managed_closes)
        self.assertFalse(status.force_flatten)
        self.assertIn("runtime_mode_not_live", status.reasons)
        self.assertIn("live_orders_not_allowed", status.reasons)
        self.assertIn("missing_exchange_credentials", status.reasons)
        stored = self.storage.get_json_state(runtime.STATUS_STATE_KEY, {})
        self.assertFalse(stored["effective_live"])
        self.assertEqual(self.storage.get_state(runtime.MODE_STATE_KEY), "dry_run")

    def test_evaluate_blocks_live_on_kill_switch_manual_recovery_and_model_disable(self):
        settings = Settings()
        settings.app.runtime_mode = "live"
        settings.app.allow_live_orders = True
        settings.exchange.symbols = ["BTC/USDT"]
        settings.exchange.api_key = SecretStr("key")
        settings.exchange.api_secret = SecretStr("secret")
        settings.exchange.api_passphrase = SecretStr("passphrase")
        runtime = AutonomyLiveRuntime(self.storage, settings=settings)
        config = runtime.build_config(requested_live=True)
        runtime.set_kill_switch(True, reason="ops_pause")
        self.storage.set_state("manual_recovery_required", "true")
        self.storage.set_state("manual_recovery_approved", "false")
        self.storage.set_state("model_degradation_status", "disabled")

        status = runtime.evaluate(
            requested_live=True,
            config=config,
        )

        self.assertFalse(status.effective_live)
        self.assertFalse(status.allow_entries)
        self.assertTrue(status.allow_managed_closes)
        self.assertTrue(status.force_flatten)
        self.assertTrue(status.kill_switch_active)
        self.assertEqual(status.kill_switch_reason, "ops_pause")
        self.assertIn("live_kill_switch_active", status.reasons)
        self.assertIn("manual_recovery_required", status.reasons)
        self.assertIn("model_degradation_disabled", status.reasons)

    def test_build_live_trader_uses_exchange_adapter_only_when_effective_live(self):
        settings = Settings()
        settings.app.runtime_mode = "live"
        settings.app.allow_live_orders = True
        settings.exchange.provider = "okx"
        settings.exchange.symbols = ["BTC/USDT"]
        settings.exchange.api_key = SecretStr("key")
        settings.exchange.api_secret = SecretStr("secret")
        settings.exchange.api_passphrase = SecretStr("passphrase")
        runtime = AutonomyLiveRuntime(
            self.storage,
            settings=settings,
            okx_exchange_adapter_cls=FakeAdapter,
        )
        config = runtime.build_config(requested_live=True)
        status = runtime.evaluate(
            requested_live=True,
            config=config,
        )

        trader = runtime.build_live_trader(status=status)

        self.assertTrue(status.effective_live)
        self.assertTrue(status.allow_entries)
        self.assertTrue(status.allow_managed_closes)
        self.assertTrue(trader.enabled)
        self.assertIsInstance(trader.exchange, FakeAdapter)
        self.assertEqual(trader.exchange.kwargs["api_key"], "key")
        self.assertEqual(trader.exchange.kwargs["api_secret"], "secret")
        self.assertEqual(trader.exchange.kwargs["api_passphrase"], "passphrase")

        dry_run_status = runtime.evaluate(
            requested_live=False,
            config=runtime.build_config(requested_live=False),
        )
        dry_run_trader = runtime.build_live_trader(status=dry_run_status)
        self.assertFalse(dry_run_status.effective_live)
        self.assertFalse(dry_run_trader.enabled)
        self.assertIsNone(dry_run_trader.exchange)

    def test_build_live_trader_keeps_close_path_enabled_during_force_flatten(self):
        settings = Settings()
        settings.app.runtime_mode = "live"
        settings.app.allow_live_orders = False
        settings.exchange.provider = "okx"
        settings.exchange.symbols = ["BTC/USDT"]
        settings.exchange.api_key = SecretStr("key")
        settings.exchange.api_secret = SecretStr("secret")
        settings.exchange.api_passphrase = SecretStr("passphrase")
        runtime = AutonomyLiveRuntime(
            self.storage,
            settings=settings,
            okx_exchange_adapter_cls=FakeAdapter,
        )
        runtime.set_kill_switch(True, reason="ops_pause")

        status = runtime.evaluate(
            requested_live=False,
            config=runtime.build_config(requested_live=False),
        )
        trader = runtime.build_live_trader(status=status)

        self.assertFalse(status.effective_live)
        self.assertFalse(status.allow_entries)
        self.assertTrue(status.allow_managed_closes)
        self.assertTrue(status.force_flatten)
        self.assertTrue(trader.enabled)
        self.assertIsInstance(trader.exchange, FakeAdapter)


if __name__ == "__main__":
    unittest.main()
