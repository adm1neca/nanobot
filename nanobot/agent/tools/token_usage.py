"""Token usage tool for querying token consumption."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.utils.token_tracker import TokenTracker


class TokenUsageTool(Tool):
    """Tool for querying token usage statistics."""

    def __init__(self, usage_dir: Path | None = None):
        """
        Initialize the token usage tool.

        Args:
            usage_dir: Path to the usage directory. Defaults to ~/.nanobot/usage.
        """
        self.usage_dir = usage_dir or Path.home() / ".nanobot" / "usage"

    @property
    def name(self) -> str:
        return "token_usage"

    @property
    def description(self) -> str:
        return "Get token usage statistics. Use this when asked about token consumption, usage totals, or model-specific usage. Returns today's usage, recent days summary, total usage, and breakdown by model."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "The scope of usage to retrieve: 'today', 'recent', 'total', 'models', or 'all'. Default is 'all'.",
                    "enum": ["today", "recent", "total", "models", "all"],
                },
                "date": {
                    "type": "string",
                    "description": "Optional specific date (YYYY-MM-DD) to get usage for. Only used when scope is 'today' or 'models'.",
                },
                "model": {
                    "type": "string",
                    "description": "Optional model name to filter by. Only used when scope is 'models'.",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Execute the token usage query."""
        scope = kwargs.get("scope", "all")
        date = kwargs.get("date")
        model_filter = kwargs.get("model")

        try:
            tracker = TokenTracker(self.usage_dir)
        except Exception as e:
            return f"Error initializing token tracker: {e}"

        try:
            if scope == "today":
                return self._get_today_usage(tracker, date)
            elif scope == "recent":
                return self._get_recent_usage(tracker)
            elif scope == "total":
                return self._get_total_usage(tracker)
            elif scope == "models":
                return self._get_model_breakdown(tracker, date, model_filter)
            else:  # "all"
                return tracker.format_summary()
        except Exception as e:
            return f"Error retrieving token usage: {e}"

    def _get_today_usage(self, tracker: TokenTracker, date: str | None = None) -> str:
        """Get today's token usage."""
        usage = tracker.get_today_usage()
        if not usage:
            return "No token usage recorded for today."

        normalized = tracker._normalize_usage(usage)
        return (
            f"Today's Token Usage:\n"
            f"  Prompt tokens: {normalized.get('prompt_tokens', 0):,}\n"
            f"  Completion tokens: {normalized.get('completion_tokens', 0):,}\n"
            f"  Total tokens: {normalized.get('total_tokens', 0):,}"
        )

    def _get_recent_usage(self, tracker: TokenTracker) -> str:
        """Get recent days' token usage."""
        recent = tracker.get_recent_days(7)
        if not recent:
            return "No token usage recorded."

        lines = ["Recent Token Usage (last 7 days):"]
        for date, usage in recent:
            normalized = tracker._normalize_usage(usage)
            total = normalized.get("total_tokens", 0)
            lines.append(f"  {date}: {total:,} tokens")
        return "\n".join(lines)

    def _get_total_usage(self, tracker: TokenTracker) -> str:
        """Get total token usage across all days."""
        totals = tracker.get_total_usage()
        if totals["total_tokens"] == 0:
            return "No token usage recorded."

        return (
            f"Total Token Usage:\n"
            f"  Prompt tokens: {totals['prompt_tokens']:,}\n"
            f"  Completion tokens: {totals['completion_tokens']:,}\n"
            f"  Total tokens: {totals['total_tokens']:,}"
        )

    def _get_model_breakdown(
        self, tracker: TokenTracker, date: str | None = None, model_filter: str | None = None
    ) -> str:
        """Get token usage breakdown by model."""
        breakdown = tracker.get_model_breakdown(date)
        if not breakdown:
            return "No model usage recorded."

        if model_filter:
            # Filter to specific model
            if model_filter in breakdown:
                stats = breakdown[model_filter]
                return (
                    f"Usage for model '{model_filter}':\n"
                    f"  Prompt: {stats.get('prompt', 0):,}\n"
                    f"  Completion: {stats.get('completion', 0):,}\n"
                    f"  Total: {stats.get('total', 0):,} tokens"
                )
            return f"No usage recorded for model '{model_filter}'."

        lines = ["Token Usage by Model:"]
        for model, stats in sorted(breakdown.items(), key=lambda x: x[1]["total"], reverse=True):
            lines.append(f"  {model}: {stats['total']:,} tokens")
        return "\n".join(lines)
