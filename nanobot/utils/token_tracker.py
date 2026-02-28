"""Token usage tracking utility for NanoBot."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class TokenTracker:
    """
    Tracks and aggregates token usage data.

    Reads/writes daily token usage to the configured usage directory.
    Supports both the legacy format (flat keys) and new format (nested structure).

    Legacy format:
    {
        "2026-02-28": {
            "prompt": 1000,
            "completion": 500,
            "total": 1500,
            "models": {
                "model-name": {"prompt": 1000, "completion": 500, "total": 1500}
            }
        }
    }
    """

    def __init__(self, usage_dir: Path | None = None):
        """Initialize the token tracker."""
        if usage_dir:
            self.usage_dir = usage_dir
        else:
            self.usage_dir = Path.home() / ".nanobot" / "usage"
        self.usage_file = self.usage_dir / "daily.json"

    def _ensure_dir(self) -> None:
        """Ensure the usage directory exists."""
        self.usage_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> dict[str, Any]:
        """Load usage data from file."""
        if not self.usage_file.exists():
            return {}

        try:
            with open(self.usage_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to load usage data, starting fresh")
            return {}

    def _save_data(self, data: dict[str, Any]) -> None:
        """Save usage data to file."""
        self._ensure_dir()
        with open(self.usage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _normalize_usage(self, day_data: dict[str, Any]) -> dict[str, int]:
        """Normalize usage data to consistent format."""
        # Check for nested "total" dict (new format)
        total_dict = day_data.get("total")
        if isinstance(total_dict, dict):
            return total_dict

        # Legacy format with flat keys
        return {
            "prompt_tokens": day_data.get("prompt", 0),
            "completion_tokens": day_data.get("completion", 0),
            "total_tokens": day_data.get("total", 0),
        }

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        date: str | None = None,
    ) -> None:
        """
        Record token usage for a specific date.

        Args:
            model: The model identifier (e.g., 'claude-opus-4-5').
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.
            total_tokens: Total tokens used.
            date: ISO date string (YYYY-MM-DD). Defaults to today.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        data = self._load_data()

        if date not in data:
            data[date] = {
                "prompt": 0,
                "completion": 0,
                "total": 0,
                "models": {},
            }

        day = data[date]

        # Update daily totals
        day["prompt"] += prompt_tokens
        day["completion"] += completion_tokens
        day["total"] += total_tokens

        # Update per-model breakdown
        if model not in day["models"]:
            day["models"][model] = {"prompt": 0, "completion": 0, "total": 0}

        model_data = day["models"][model]
        model_data["prompt"] += prompt_tokens
        model_data["completion"] += completion_tokens
        model_data["total"] += total_tokens

        self._save_data(data)
        logger.debug("Recorded {} tokens for model {} on {}", total_tokens, model, date)

    def get_today_usage(self) -> dict[str, Any] | None:
        """Get token usage for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        data = self._load_data()
        return data.get(today)

    def get_recent_days(self, days: int = 7) -> list[tuple[str, dict[str, Any]]]:
        """
        Get usage data for the last N days.

        Returns:
            List of (date, usage_data) tuples, sorted by date descending.
        """
        data = self._load_data()
        sorted_dates = sorted(data.keys(), reverse=True)[:days]
        return [(date, data[date]) for date in sorted_dates]

    def get_total_usage(self) -> dict[str, int]:
        """
        Get total usage across all days.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens.
        """
        data = self._load_data()
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for day_data in data.values():
            normalized = self._normalize_usage(day_data)
            totals["prompt_tokens"] += normalized.get("prompt_tokens", 0)
            totals["completion_tokens"] += normalized.get("completion_tokens", 0)
            totals["total_tokens"] += normalized.get("total_tokens", 0)

        return totals

    def get_model_breakdown(self, date: str | None = None) -> dict[str, dict[str, int]]:
        """
        Get token usage breakdown by model.

        Args:
            date: Optional date string. If None, returns breakdown for all days.

        Returns:
            Dict mapping model names to their usage stats.
        """
        data = self._load_data()

        if date:
            day_data = data.get(date, {})
            return day_data.get("models", {})

        # Aggregate across all days
        breakdown: dict[str, dict[str, int]] = {}
        for day_data in data.values():
            for model, model_data in day_data.get("models", {}).items():
                if model not in breakdown:
                    breakdown[model] = {"prompt": 0, "completion": 0, "total": 0}
                breakdown[model]["prompt"] += model_data.get("prompt", 0)
                breakdown[model]["completion"] += model_data.get("completion", 0)
                breakdown[model]["total"] += model_data.get("total", 0)

        return breakdown

    def format_usage(self, usage: dict[str, Any]) -> str:
        """Format usage data for display."""
        normalized = self._normalize_usage(usage)
        lines = [
            f"Prompt tokens: {normalized.get('prompt_tokens', 0):,}",
            f"Completion tokens: {normalized.get('completion_tokens', 0):,}",
            f"Total tokens: {normalized.get('total_tokens', 0):,}",
        ]
        return "\n".join(lines)

    def format_summary(self) -> str:
        """
        Format a complete token usage summary.

        Returns:
            Formatted string with today's usage, recent days, and totals.
        """
        lines = ["Token Usage Overview", "=" * 20]

        # Today's usage
        today = datetime.now().strftime("%Y-%m-%d")
        today_usage = self.get_today_usage()
        if today_usage:
            normalized = self._normalize_usage(today_usage)
            lines.append(f"\nToday ({today}):")
            lines.append(f"  Prompt tokens: {normalized.get('prompt_tokens', 0):,}")
            lines.append(f"  Completion tokens: {normalized.get('completion_tokens', 0):,}")
            lines.append(f"  Total tokens: {normalized.get('total_tokens', 0):,}")
        else:
            lines.append(f"\nToday ({today}): No usage yet")

        # Recent days (excluding today)
        recent = self.get_recent_days(8)
        if len(recent) > 1:
            lines.append("\nRecent Days:")
            for date, usage in recent:
                if date != today:
                    normalized = self._normalize_usage(usage)
                    total = normalized.get("total_tokens", 0)
                    lines.append(f"  {date}: {total:,} tokens")

        # Total usage
        totals = self.get_total_usage()
        lines.append("\nTotal Usage:")
        lines.append(f"  Prompt tokens: {totals['prompt_tokens']:,}")
        lines.append(f"  Completion tokens: {totals['completion_tokens']:,}")
        lines.append(f"  Total tokens: {totals['total_tokens']:,}")

        # Model breakdown
        breakdown = self.get_model_breakdown()
        if breakdown:
            lines.append("\nBy Model:")
            for model, stats in sorted(breakdown.items(), key=lambda x: x[1]["total"], reverse=True):
                lines.append(f"  {model}: {stats['total']:,} tokens")

        return "\n".join(lines)
