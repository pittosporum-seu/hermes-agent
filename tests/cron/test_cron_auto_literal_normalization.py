"""Literal ``"auto"`` model/provider normalization for cron jobs.

Background: jobs created via ``cronjob create model=auto provider=auto``
persist the sentinel string verbatim. Before this fix the run path treated
the non-empty string as a pin:

  - ``model = job.get("model") or ...`` kept ``"auto"`` and skipped the
    config-default block (``if not job.get("model")``), so the literal
    ``"auto"`` reached the provider as a model name. The API rejects it
    (HTTP 401 "Model auto is not supported") and the fallback chain
    silently reroutes the run to the first fallback_providers entry —
    spending an unrelated provider's balance on every tick.
  - ``"requested": job.get("provider") or ...`` passed ``"auto"`` past the
    cron.model_provider fleet default.

The fix normalizes the literal ``"auto"`` (case-insensitive) on both axes
to "unpinned" so the default resolution (cron.model > config model.default
> HERMES_MODEL) applies — mirroring the auxiliary-path normalization in
agent/auxiliary_client.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import _normalize_auto_pin, run_job


def _base_job(**overrides):
    job = {
        "id": "auto-norm-test",
        "name": "auto norm test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "model_snapshot": None,
        "provider_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _write_config(tmp_path, text):
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")


def _run_job(job, tmp_path, resolve_spy):
    """Drive run_job with resolve_runtime_provider spied and AIAgent mocked.

    Returns (success, agent_kwargs_or_None).
    """
    fake_db = MagicMock()

    def _fake_resolve(**kwargs):
        resolve_spy.append(kwargs)
        return {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": "test-provider",
            "api_mode": "chat_completions",
        }

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             side_effect=_fake_resolve,
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent

        success, _output, _final_response, _error = run_job(job)
        agent_kwargs = mock_agent_cls.call_args.kwargs if mock_agent_cls.called else None

    return success, agent_kwargs


class TestNormalizeAutoPin:
    def test_literal_auto_maps_to_none(self):
        assert _normalize_auto_pin("auto") is None

    def test_case_and_whitespace_insensitive(self):
        assert _normalize_auto_pin("AUTO") is None
        assert _normalize_auto_pin(" Auto ") is None

    def test_real_pins_pass_through(self):
        assert _normalize_auto_pin("deepseek-v4-flash") == "deepseek-v4-flash"
        assert _normalize_auto_pin("automatic-model") == "automatic-model"
        assert _normalize_auto_pin(None) is None
        assert _normalize_auto_pin("") == ""


class TestPreflightAutoNormalization:
    """The pre-dispatch provider-key probe mirrors run_job's resolution and
    must see the same normalized values — otherwise it resolves against the
    literal ``"auto"`` while the actual run resolves against the config
    default, so the probe can miss a genuinely missing key (or falsely
    block a run that the fallback chain would rescue)."""

    def test_preflight_sees_fleet_provider_not_literal_auto(self, tmp_path):
        from cron.scheduler import _preflight_check_provider_key

        cfg = {
            "cron": {"model": "fleet-model", "model_provider": "fleet-provider"},
        }
        job = _base_job(model="auto", provider="auto")
        calls = []

        def _fake_resolve(**kwargs):
            calls.append(kwargs)
            return {"provider": "fleet-provider"}

        with patch("cron.scheduler.get_fallback_chain", return_value=[]), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 side_effect=_fake_resolve,
             ):
            result = _preflight_check_provider_key(job, cfg)

        assert result is None, f"preflight should pass, got: {result}"
        assert calls, "resolve_runtime_provider was not probed"
        assert calls[0]["requested"] == "fleet-provider", (
            f"preflight must normalize provider='auto' before probing; "
            f"got requested={calls[0]['requested']!r}"
        )


class TestLiteralAutoNormalization:
    def test_model_auto_resolves_to_config_default(self, tmp_path):
        """model='auto' must NOT reach the wire; config default applies."""
        _write_config(
            tmp_path,
            "model:\n  default: deepseek-v4-flash\n  provider: opencode-go\n",
        )
        resolve_spy = []
        success, agent_kwargs = _run_job(
            _base_job(model="auto", provider="auto"), tmp_path, resolve_spy
        )
        assert success, "job with model=auto should run after normalization"
        assert agent_kwargs is not None, "AIAgent should have been constructed"
        assert agent_kwargs["model"] == "deepseek-v4-flash", (
            f"literal 'auto' leaked into the model slot: {agent_kwargs['model']!r}"
        )
        # Every resolution probe (the pre-dispatch provider-key preflight
        # and the actual run resolution) must never receive the literal
        # sentinel. The preflight probe reads the real user config, so its
        # resolved model may legitimately differ from this test's config
        # default — only the final run resolution must match it.
        assert resolve_spy, "resolve_runtime_provider was not called"
        for i, call in enumerate(resolve_spy):
            target = str(call.get("target_model") or "").strip().lower()
            assert target != "auto", (
                f"resolve call #{i} got literal 'auto' in the model slot: "
                f"{call['target_model']!r}"
            )
        assert resolve_spy[-1]["target_model"] == "deepseek-v4-flash", (
            "the final run resolution must use the config default, got "
            f"{resolve_spy[-1]['target_model']!r}"
        )

    def test_provider_auto_does_not_short_circuit_fleet_default(self, tmp_path):
        """provider='auto' must not shadow cron.model_provider."""
        _write_config(
            tmp_path,
            "model:\n  default: deepseek-v4-flash\n"
            "cron:\n  model: fleet-model\n  model_provider: fleet-provider\n",
        )
        resolve_spy = []
        success, agent_kwargs = _run_job(
            _base_job(model="auto", provider="auto"), tmp_path, resolve_spy
        )
        assert success
        assert resolve_spy, "resolve_runtime_provider was not called"
        assert resolve_spy[0]["requested"] == "fleet-provider", (
            f"provider='auto' should normalize to unpinned so cron.model_provider "
            f"applies; got requested={resolve_spy[0]['requested']!r}"
        )
        assert agent_kwargs["model"] == "fleet-model"

    def test_model_auto_case_insensitive(self, tmp_path):
        """'AUTO' / ' Auto ' normalize the same way."""
        _write_config(tmp_path, "model:\n  default: resolved-model\n")
        resolve_spy = []
        success, agent_kwargs = _run_job(
            _base_job(model=" AUTO ", provider="Auto"), tmp_path, resolve_spy
        )
        assert success
        assert agent_kwargs["model"] == "resolved-model"

    def test_explicit_pin_still_wins(self, tmp_path):
        """A real (non-'auto') per-job pin is untouched."""
        _write_config(tmp_path, "model:\n  default: config-default\n")
        resolve_spy = []
        success, agent_kwargs = _run_job(
            _base_job(model="pinned-model", provider="pinned-provider"),
            tmp_path,
            resolve_spy,
        )
        assert success
        assert agent_kwargs["model"] == "pinned-model"
        assert resolve_spy[0]["requested"] == "pinned-provider"

    def test_unpinned_job_still_follows_config_default(self, tmp_path):
        """Regression: None-valued axes keep the pre-existing behavior."""
        _write_config(tmp_path, "model:\n  default: config-default\n")
        resolve_spy = []
        success, agent_kwargs = _run_job(_base_job(), tmp_path, resolve_spy)
        assert success
        assert agent_kwargs["model"] == "config-default"
