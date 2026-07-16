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


if __name__ == "__main__":
    unittest.main()
