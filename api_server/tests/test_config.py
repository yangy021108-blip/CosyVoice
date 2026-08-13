from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from api_server.config import Settings


class SettingsTest(unittest.TestCase):
    def test_default_binding_is_loopback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertIsNone(settings.api_key)
        self.assertEqual(settings.default_seed, 2)
        self.assertTrue(settings.quality_check_enabled)
        self.assertEqual(settings.quality_max_retries, 2)
        self.assertEqual(settings.stream_queue_size, 4)

    def test_non_loopback_binding_without_auth_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"COSYVOICE_HOST": "0.0.0.0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "COSYVOICE_API_KEY"):
                Settings.from_env()

    def test_non_loopback_binding_with_api_key_is_allowed(self) -> None:
        with patch.dict(
            os.environ,
            {
                "COSYVOICE_HOST": "0.0.0.0",
                "COSYVOICE_API_KEY": "test-key",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.api_key, "test-key")

    def test_explicit_unauthenticated_opt_in_is_allowed(self) -> None:
        with patch.dict(
            os.environ,
            {
                "COSYVOICE_HOST": "0.0.0.0",
                "COSYVOICE_ALLOW_UNAUTHENTICATED": "true",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertTrue(settings.allow_unauthenticated)

    def test_quality_settings_are_read_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "COSYVOICE_DEFAULT_SEED": "17",
                "COSYVOICE_QUALITY_CHECK_ENABLED": "false",
                "COSYVOICE_QUALITY_MAX_RETRIES": "3",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.default_seed, 17)
        self.assertFalse(settings.quality_check_enabled)
        self.assertEqual(settings.quality_max_retries, 3)

    def test_seed_above_uint32_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"COSYVOICE_DEFAULT_SEED": str(2**32)},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "COSYVOICE_DEFAULT_SEED"):
                Settings.from_env()

    def test_stream_queue_size_is_read_and_validated(self) -> None:
        with patch.dict(
            os.environ,
            {"COSYVOICE_STREAM_QUEUE_SIZE": "8"},
            clear=True,
        ):
            self.assertEqual(Settings.from_env().stream_queue_size, 8)
        with patch.dict(
            os.environ,
            {"COSYVOICE_STREAM_QUEUE_SIZE": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError, "COSYVOICE_STREAM_QUEUE_SIZE"
            ):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
