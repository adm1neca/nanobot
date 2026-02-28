"""Tests for per-subagent model override via spawn(model=...)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_provider(default_model: str = "fast-model") -> MagicMock:
    p = MagicMock()
    p.get_default_model.return_value = default_model
    p.chat = AsyncMock(return_value=MagicMock(has_tool_calls=False, content="done"))
    return p


def _make_manager(provider=None, model_providers=None):
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    provider = provider or _make_provider()
    bus = MessageBus()
    return SubagentManager(
        provider=provider,
        workspace=MagicMock(),
        bus=bus,
        model=provider.get_default_model(),
        model_providers=model_providers,
    )


# ---------------------------------------------------------------------------
# _resolve_provider
# ---------------------------------------------------------------------------

class TestResolveProvider:
    def test_no_overrides_returns_default(self):
        prov = _make_provider("fast-model")
        mgr = _make_manager(provider=prov)

        resolved_prov, resolved_model = mgr._resolve_provider("fast-model")
        assert resolved_prov is prov
        assert resolved_model == "fast-model"

    def test_exact_match(self):
        prov = _make_provider("fast-model")
        opus_prov = _make_provider("claude-opus-4-6")
        mgr = _make_manager(provider=prov, model_providers={"claude-opus-4-6": opus_prov})

        resolved_prov, resolved_model = mgr._resolve_provider("claude-opus-4-6")
        assert resolved_prov is opus_prov
        assert resolved_model == "claude-opus-4-6"

    def test_prefix_match(self):
        prov = _make_provider("fast-model")
        opus_prov = _make_provider("claude-opus-4-6")
        mgr = _make_manager(provider=prov, model_providers={"claude-": opus_prov})

        resolved_prov, resolved_model = mgr._resolve_provider("claude-opus-4-6")
        assert resolved_prov is opus_prov
        assert resolved_model == "claude-opus-4-6"

    def test_no_match_falls_back_to_default(self):
        prov = _make_provider("fast-model")
        opus_prov = _make_provider("claude-opus-4-6")
        mgr = _make_manager(provider=prov, model_providers={"claude-": opus_prov})

        resolved_prov, resolved_model = mgr._resolve_provider("gpt-4o")
        assert resolved_prov is prov
        assert resolved_model == "gpt-4o"

    def test_exact_beats_prefix(self):
        prov = _make_provider("fast-model")
        exact_prov = _make_provider("claude-opus-4-6")
        prefix_prov = _make_provider("claude-opus-4-6")
        mgr = _make_manager(
            provider=prov,
            model_providers={
                "claude-opus-4-6": exact_prov,
                "claude-": prefix_prov,
            },
        )
        resolved_prov, _ = mgr._resolve_provider("claude-opus-4-6")
        assert resolved_prov is exact_prov


# ---------------------------------------------------------------------------
# spawn() routes to the right provider
# ---------------------------------------------------------------------------

class TestSpawnModelRouting:
    @pytest.mark.asyncio
    async def test_spawn_without_model_uses_default(self):
        default_prov = _make_provider("fast-model")
        opus_prov = _make_provider("claude-opus-4-6")
        mgr = _make_manager(provider=default_prov, model_providers={"claude-": opus_prov})

        with patch.object(mgr, "_run_subagent", new_callable=AsyncMock) as mock_run:
            await mgr.spawn(task="quick task")
            _, _, _, _, used_prov, used_model = mock_run.call_args.args
            assert used_prov is default_prov
            assert used_model == "fast-model"

    @pytest.mark.asyncio
    async def test_spawn_with_model_override_uses_matched_provider(self):
        default_prov = _make_provider("fast-model")
        opus_prov = _make_provider("claude-opus-4-6")
        mgr = _make_manager(provider=default_prov, model_providers={"claude-": opus_prov})

        with patch.object(mgr, "_run_subagent", new_callable=AsyncMock) as mock_run:
            await mgr.spawn(task="hard task", model="claude-opus-4-6")
            _, _, _, _, used_prov, used_model = mock_run.call_args.args
            assert used_prov is opus_prov
            assert used_model == "claude-opus-4-6"

    @pytest.mark.asyncio
    async def test_spawn_with_unknown_model_falls_back(self):
        default_prov = _make_provider("fast-model")
        opus_prov = _make_provider("claude-opus-4-6")
        mgr = _make_manager(provider=default_prov, model_providers={"claude-": opus_prov})

        with patch.object(mgr, "_run_subagent", new_callable=AsyncMock) as mock_run:
            await mgr.spawn(task="some task", model="unknown-model-99")
            _, _, _, _, used_prov, used_model = mock_run.call_args.args
            assert used_prov is default_prov
            assert used_model == "unknown-model-99"

    @pytest.mark.asyncio
    async def test_spawn_return_message_includes_label(self):
        mgr = _make_manager()
        with patch.object(mgr, "_run_subagent", new_callable=AsyncMock):
            result = await mgr.spawn(task="do stuff", label="my-label")
        assert "my-label" in result

    @pytest.mark.asyncio
    async def test_openrouter_anthropic_prefix_routes_to_openrouter(self):
        """Verify the real-world setup: openrouter keyed on 'anthropic/' prefix."""
        default_prov = _make_provider("minimax-m2.5:cloud")
        openrouter_prov = _make_provider("anthropic/claude-opus-4-6")
        mgr = _make_manager(
            provider=default_prov,
            model_providers={"anthropic/": openrouter_prov},
        )

        with patch.object(mgr, "_run_subagent", new_callable=AsyncMock) as mock_run:
            await mgr.spawn(task="plan quarterly goals", model="anthropic/claude-opus-4-6")
            _, _, _, _, used_prov, used_model = mock_run.call_args.args
            assert used_prov is openrouter_prov
            assert used_model == "anthropic/claude-opus-4-6"

    @pytest.mark.asyncio
    async def test_openrouter_prefix_does_not_match_non_anthropic_models(self):
        """'anthropic/' prefix must not catch e.g. 'openai/gpt-4o'."""
        default_prov = _make_provider("minimax-m2.5:cloud")
        openrouter_prov = _make_provider("anthropic/claude-opus-4-6")
        mgr = _make_manager(
            provider=default_prov,
            model_providers={"anthropic/": openrouter_prov},
        )

        with patch.object(mgr, "_run_subagent", new_callable=AsyncMock) as mock_run:
            await mgr.spawn(task="quick task", model="openai/gpt-4o")
            _, _, _, _, used_prov, used_model = mock_run.call_args.args
            assert used_prov is default_prov
            assert used_model == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# SpawnTool exposes the model parameter
# ---------------------------------------------------------------------------

class TestSpawnToolSchema:
    def test_model_param_in_schema(self):
        from nanobot.agent.tools.spawn import SpawnTool

        tool = SpawnTool(manager=MagicMock())
        props = tool.parameters["properties"]
        assert "model" in props
        assert props["model"]["type"] == "string"

    def test_model_not_required(self):
        from nanobot.agent.tools.spawn import SpawnTool

        tool = SpawnTool(manager=MagicMock())
        assert "model" not in tool.parameters.get("required", [])

    @pytest.mark.asyncio
    async def test_execute_passes_model_to_manager(self):
        from nanobot.agent.tools.spawn import SpawnTool

        mock_mgr = MagicMock()
        mock_mgr.spawn = AsyncMock(return_value="started")
        tool = SpawnTool(manager=mock_mgr)

        await tool.execute(task="hard task", model="claude-opus-4-6")

        mock_mgr.spawn.assert_called_once()
        call_kwargs = mock_mgr.spawn.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-6"

    @pytest.mark.asyncio
    async def test_execute_passes_none_model_when_omitted(self):
        from nanobot.agent.tools.spawn import SpawnTool

        mock_mgr = MagicMock()
        mock_mgr.spawn = AsyncMock(return_value="started")
        tool = SpawnTool(manager=mock_mgr)

        await tool.execute(task="easy task")

        call_kwargs = mock_mgr.spawn.call_args.kwargs
        assert call_kwargs["model"] is None


# ---------------------------------------------------------------------------
# Token usage is recorded for subagent LLM calls
# ---------------------------------------------------------------------------

class TestSubagentTokenRecording:
    def test_record_token_usage_calls_tracker(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from unittest.mock import patch

        mgr = SubagentManager(
            provider=_make_provider(),
            workspace=MagicMock(),
            bus=MessageBus(),
        )

        response = MagicMock()
        response.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        with patch("nanobot.agent.subagent.TokenTracker") as MockTracker:
            mgr._record_token_usage(response, "anthropic/claude-opus-4-6")

        MockTracker.return_value.record_usage.assert_called_once_with(
            model="anthropic/claude-opus-4-6",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

    def test_record_token_usage_skips_when_no_usage(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from unittest.mock import patch

        mgr = SubagentManager(
            provider=_make_provider(),
            workspace=MagicMock(),
            bus=MessageBus(),
        )

        response = MagicMock()
        response.usage = None

        with patch("nanobot.agent.subagent.TokenTracker") as MockTracker:
            mgr._record_token_usage(response, "anthropic/claude-opus-4-6")

        MockTracker.assert_not_called()

    def test_record_token_usage_swallows_errors(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from unittest.mock import patch

        mgr = SubagentManager(
            provider=_make_provider(),
            workspace=MagicMock(),
            bus=MessageBus(),
        )

        response = MagicMock()
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        with patch("nanobot.agent.subagent.TokenTracker", side_effect=Exception("disk full")):
            # Must not raise
            mgr._record_token_usage(response, "some-model")


# ---------------------------------------------------------------------------
# AgentLoop propagates model_providers to SubagentManager
# ---------------------------------------------------------------------------

class TestAgentLoopModelProviders:
    def test_model_providers_passed_to_subagent_manager(self):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = _make_provider()
        opus_prov = _make_provider("claude-opus-4-6")
        model_providers = {"claude-": opus_prov}
        workspace = MagicMock()
        workspace.__truediv__ = MagicMock(return_value=MagicMock())

        with patch("nanobot.agent.loop.ContextBuilder"), \
             patch("nanobot.agent.loop.SessionManager"):
            loop = AgentLoop(
                bus=bus,
                provider=provider,
                workspace=workspace,
                model_providers=model_providers,
            )

        assert loop.subagents._model_providers is model_providers

    def test_no_model_providers_defaults_to_empty(self):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = _make_provider()
        workspace = MagicMock()
        workspace.__truediv__ = MagicMock(return_value=MagicMock())

        with patch("nanobot.agent.loop.ContextBuilder"), \
             patch("nanobot.agent.loop.SessionManager"):
            loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

        assert loop.subagents._model_providers == {}
