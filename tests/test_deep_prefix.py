"""Tests for the deep: prefix — deterministic routing to the capable model."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage


def _make_loop(model_providers=None):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "fast-model"
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

    # Configure session mock so integer operations in _process_message work.
    mock_session = MagicMock()
    mock_session.messages = []
    mock_session.last_consolidated = 0
    mock_session.key = "test:c1"
    mock_session.get_history.return_value = []
    loop.sessions.get_or_create.return_value = mock_session

    return loop, bus


def _inbound(content: str) -> InboundMessage:
    return InboundMessage(channel="test", sender_id="u1", chat_id="c1", content=content)


class TestDeepPrefix:
    @pytest.mark.asyncio
    async def test_deep_spawns_subagent_with_deep_model(self):
        """deep: bypasses the LLM and spawns directly to the override model."""
        opus_prov = MagicMock()
        opus_prov.default_model = "anthropic/claude-opus-4-6"
        loop, bus = _make_loop(model_providers={"anthropic/": opus_prov})

        loop.subagents.spawn = AsyncMock(return_value="Subagent [plan] started.")

        msg = _inbound("deep: draft my Q2 strategy")
        response = await loop._process_message(msg)

        loop.subagents.spawn.assert_called_once()
        kwargs = loop.subagents.spawn.call_args.kwargs
        assert kwargs["task"] == "draft my Q2 strategy"
        assert kwargs["model"] == "anthropic/claude-opus-4-6"
        assert kwargs["origin_channel"] == "test"
        assert kwargs["origin_chat_id"] == "c1"

    @pytest.mark.asyncio
    async def test_deep_case_insensitive(self):
        """DEEP:, Deep:, deep: all trigger routing."""
        opus_prov = MagicMock()
        opus_prov.default_model = "anthropic/claude-opus-4-6"
        loop, _ = _make_loop(model_providers={"anthropic/": opus_prov})
        loop.subagents.spawn = AsyncMock(return_value="started")

        for prefix in ("DEEP: task", "Deep: task", "deep: task", "deep:task"):
            loop.subagents.spawn.reset_mock()
            await loop._process_message(_inbound(prefix))
            loop.subagents.spawn.assert_called_once()

    @pytest.mark.asyncio
    async def test_deep_empty_task_returns_usage(self):
        """deep: with no task returns a usage hint, does not spawn."""
        loop, _ = _make_loop()
        loop.subagents.spawn = AsyncMock()

        response = await loop._process_message(_inbound("deep:"))
        assert response is not None
        assert "Usage" in response.content
        loop.subagents.spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_deep_falls_back_to_default_model_when_no_overrides(self):
        """When no model_providers are configured, deep: uses the main model."""
        loop, _ = _make_loop(model_providers={})
        loop.subagents.spawn = AsyncMock(return_value="started")

        await loop._process_message(_inbound("deep: analyze this"))

        kwargs = loop.subagents.spawn.call_args.kwargs
        assert kwargs["model"] == "fast-model"

    @pytest.mark.asyncio
    async def test_deep_does_not_call_llm(self):
        """deep: must not reach the main LLM at all."""
        opus_prov = MagicMock()
        opus_prov.default_model = "anthropic/claude-opus-4-6"
        loop, _ = _make_loop(model_providers={"anthropic/": opus_prov})
        loop.subagents.spawn = AsyncMock(return_value="started")

        with patch.object(loop, "_run_agent_loop", new_callable=AsyncMock) as mock_llm:
            await loop._process_message(_inbound("deep: write a plan"))
            mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_message_still_uses_llm(self):
        """Non-deep: messages are unaffected and go through the LLM."""
        loop, _ = _make_loop()

        with patch.object(
            loop, "_run_agent_loop", new_callable=AsyncMock,
            return_value=("reply", [], [])
        ) as mock_llm:
            await loop._process_message(_inbound("hello"))
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_deep_help_text_mentions_command(self):
        """/help output lists the deep: command."""
        loop, bus = _make_loop()
        msg = _inbound("/help")
        response = await loop._process_message(msg)
        assert response is not None
        assert "deep:" in response.content
