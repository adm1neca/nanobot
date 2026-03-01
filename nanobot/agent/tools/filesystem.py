"""File system tools: read, write, edit, list."""

import difflib
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


_MEMORY_WRITE_BLOCKED = (
    "Error: memory/MEMORY.md is managed by the memory consolidation system and cannot be "
    "written to directly. To capture an idea, note, or task, create a NEW file in "
    "00-Inbox/ instead (e.g., 00-Inbox/YYYY-MM-DD-HHmm-<slug>.md)."
)


def _resolve_path(
    path: str, workspace: Path | None = None, allowed_dir: Path | None = None
) -> Path:
    """Resolve path against workspace (if relative) and enforce directory restriction."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if allowed_dir:
        try:
            resolved.relative_to(allowed_dir.resolve())
        except ValueError:
            raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


def _is_memory_file(path: Path) -> bool:
    """Return True if the path is the long-term memory file (write-protected from agent tools)."""
    return path.name == "MEMORY.md" and path.parent.name == "memory"


class _FsTool(Tool):
    """Shared base for filesystem tools — common init and path resolution."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    def _resolve(self, path: str) -> Path:
        return _resolve_path(path, self._workspace, self._allowed_dir)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class ReadFileTool(_FsTool):
    """Read file contents with optional line-based pagination."""

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Returns numbered lines. "
            "Use offset and limit to paginate through large files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read"},
                "offset": {"type": "integer", "description": "Line number to start at (0-based)", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to return", "default": self._DEFAULT_LIMIT},
            },
            "required": ["path"],
        }

    async def execute(self, path: str, offset: int = 0, limit: int = _DEFAULT_LIMIT, **kwargs: Any) -> str:
        try:
            fp = self._resolve(path)
            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")
            lines = content.split("\n")

            if offset >= len(lines):
                return f"Error: offset {offset} is beyond file length {len(lines)}"

            # Always include the offset line and some context after
            window = lines[offset : offset + limit]
            visible_start = offset

            # Adjust offset downward to show at least 2 lines of context when possible
            if offset > 1:
                visible_start = max(0, offset - 2)
                window = lines[visible_start : offset + limit]

            # Truncate extremely long lines for readability, but include them in the count
            truncated = []
            for line in window:
                if len(line) > 500:
                    truncated.append(line[:500] + " [...]")
                else:
                    truncated.append(line)

            result = "\n".join(f"{visible_start + i + 1:6}  {line}" for i, line in enumerate(truncated))
            total = len(lines)
            shown = len(window)
            next_offset = offset + limit

            extra = ""
            if next_offset < total:
                extra = f"\n\t\t[{next_offset} more lines available, use offset={next_offset}]"
            elif offset > 0:
                extra = "\n\t\t[beginning of file]"

            final_extra = extra
            if uses_crlf:
                final_extra = (extra + " (note: file uses CRLF line endings)") if extra else " (note: file uses CRLF line endings)"

            return f"{result}\n\t\t[{shown}/{total} lines shown]{final_extra}"
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class WriteFileTool(_FsTool):
    """Tool to write content to a file (creates or overwrites)."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file (creates or overwrites). Use it to create new files or replace existing ones entirely."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if _is_memory_file(file_path):
                return _MEMORY_WRITE_BLOCKED
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {file_path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    """Find exact match in content, handling common whitespace issues."""
    # Direct match first
    count = content.count(old_text)
    if count > 0:
        return old_text, count

    # Try normalized whitespace
    normalized = old_text.strip()
    if not normalized:
        return None, 0
    normalized_count = content.count(normalized)
    if normalized_count > 0:
        return normalized, normalized_count

    return None, 0


class EditFileTool(_FsTool):
    """Tool to edit a specific portion of a file."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file by replacing exact text. Provides at least 2 lines of context for safety."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "The text to find and replace"},
                "new_text": {"type": "string", "description": "The text to replace with"},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default false)",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    def _not_found_msg(self, old_text: str, content: str, path: str) -> str:
        """Build a helpful error message when old_text isn't found."""
        # Show up to 40 lines of context centered around the search term
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if old_text in line:
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                context = "\n".join(f"  {i+1:3} | {lines[r]}" for r in range(start, end))
                return f"Error: Text not found in {path}. Did you mean:\n\n{context}\n\nNote: All indentation must match exactly."
        return f"Error: Text not found in {path}. Use read_file to see the exact content."

    async def execute(
        self, path: str, old_text: str, new_text: str,
        replace_all: bool = False, **kwargs: Any,
    ) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if _is_memory_file(file_path):
                return _MEMORY_WRITE_BLOCKED
            if not file_path.exists():
                return f"Error: File not found: {path}"

            raw = file_path.read_bytes()
            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")
            match, count = _find_match(content, old_text.replace("\r\n", "\n"))

            if match is None:
                return self._not_found_msg(old_text, content, path)
            if count > 1 and not replace_all:
                return (
                    f"Warning: old_text appears {count} times. "
                    "Provide more context to make it unique, or set replace_all=true."
                )

            norm_new = new_text.replace("\r\n", "\n")
            new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)
            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")

            file_path.write_bytes(new_content.encode("utf-8"))
            return f"Successfully edited {file_path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {e}"


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------

class ListFilesTool(_FsTool):
    """Tool to list files in a directory (non-recursive, sorted by mtime)."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List files in a directory (non-recursive), sorted by most recently modified."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            dir_path = self._resolve(path)
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"
            # Sort by mtime, newest first
            entries = sorted(dir_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if not entries:
                return f"Directory is empty: {path}"
            lines = []
            for p in entries:
                suffix = "/" if p.is_dir() else ""
                size = f" ({p.stat().st_size} bytes)" if p.is_file() else ""
                lines.append(f"  {p.name}{suffix}{size}")
            return "\n".join(lines)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"