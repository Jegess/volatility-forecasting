"""Tests for theta.config module -- config loading and validation."""

import pytest
from pathlib import Path


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_toml_returns_app_config(self, sample_config_path):
        """load_config with valid TOML returns AppConfig with correct values."""
        from theta.config import load_config, AppConfig

        config = load_config(sample_config_path)

        assert isinstance(config, AppConfig)
        assert config.symbols.universe == ["SPY", "QQQ", "AAPL"]
        assert config.api.base_url == "http://127.0.0.1:25503/v3"
        assert config.api.concurrency == 2
        assert config.dates.start_date == "2020-01-01"
        assert config.dates.end_date == "2024-12-31"

    def test_load_missing_file_raises_error(self, tmp_path):
        """load_config with missing file raises FileNotFoundError."""
        from theta.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.toml")

    def test_load_empty_toml_uses_defaults(self, tmp_path):
        """Default config (empty TOML) has 147 symbols, correct base_url, concurrency=4."""
        from theta.config import load_config

        empty_toml = tmp_path / "empty.toml"
        empty_toml.write_text("", encoding="utf-8")

        config = load_config(empty_toml)

        assert len(config.symbols.universe) == 147
        assert config.api.base_url == "http://127.0.0.1:25503/v3"
        assert config.api.concurrency == 4

    def test_symbols_can_be_overridden(self, tmp_path):
        """symbols.universe can be overridden via TOML."""
        from theta.config import load_config

        toml_content = '[symbols]\nuniverse = ["TSLA", "NVDA"]\n'
        config_file = tmp_path / "custom.toml"
        config_file.write_text(toml_content, encoding="utf-8")

        config = load_config(config_file)

        assert config.symbols.universe == ["TSLA", "NVDA"]

    def test_dates_parse_correctly(self, sample_config_path):
        """dates.start_date and end_date parse correctly from TOML."""
        from theta.config import load_config

        config = load_config(sample_config_path)

        assert config.dates.start_date == "2020-01-01"
        assert config.dates.end_date == "2024-12-31"


class TestConfigValidation:
    """Tests for config field validation."""

    def test_rejects_concurrency_zero(self):
        """Config rejects concurrency=0 (validation error)."""
        from theta.config import ApiConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ApiConfig(concurrency=0)

    def test_rejects_concurrency_above_max(self):
        """Config rejects concurrency=9 (above max 8)."""
        from theta.config import ApiConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ApiConfig(concurrency=9)

    def test_accepts_valid_concurrency(self):
        """Config accepts valid concurrency values (1-8)."""
        from theta.config import ApiConfig

        for val in [1, 4, 8]:
            api = ApiConfig(concurrency=val)
            assert api.concurrency == val


class TestDefaultConfig:
    """Tests for default AppConfig values."""

    def test_default_has_full_universe(self, default_config):
        """Default config has exactly 147 symbols."""
        assert len(default_config.symbols.universe) == 147

    def test_default_symbols_start_with_spy_qqq(self, default_config):
        """Default symbols list starts with SPY and QQQ."""
        assert default_config.symbols.universe[0] == "SPY"
        assert default_config.symbols.universe[1] == "QQQ"

    def test_default_api_settings(self, default_config):
        """Default API settings are correct."""
        assert default_config.api.base_url == "http://127.0.0.1:25503/v3"
        assert default_config.api.concurrency == 4
        assert default_config.api.timeout_default == 60.0
        assert default_config.api.timeout_large == 300.0
        assert default_config.api.large_symbols == ["SPY", "QQQ", "IWM"]

    def test_default_date_range(self, default_config):
        """Default date range is 2020-01-01 to 2026-03-09."""
        assert default_config.dates.start_date == "2020-01-01"
        assert default_config.dates.end_date == "2026-03-09"


class TestIntegration:
    """Integration tests using the real pipeline.toml."""

    @pytest.mark.skipif(
        not Path("pipeline.toml").exists(),
        reason="pipeline.toml not in cwd",
    )
    def test_real_pipeline_toml_loads(self):
        """Real pipeline.toml loads successfully with expected values."""
        from theta.config import load_config

        config = load_config(Path("pipeline.toml"))

        assert len(config.symbols.universe) == 194
        assert config.api.base_url == "http://127.0.0.1:25503/v3"
        assert config.api.concurrency == 4
