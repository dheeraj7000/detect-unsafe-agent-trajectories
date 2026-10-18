"""Dockerized tool sandbox interface with mock implementations.

The sandbox provides safe, reproducible tool execution (email, files, web,
code, mock APIs). In production this runs inside Docker; here we provide
in-process mock backends so the harness is runnable without external deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SandboxResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)


class SandboxBackend:
    """Abstract sandbox backend."""

    def run(self, tool: str, arguments: Dict[str, Any]) -> SandboxResult:
        raise NotImplementedError


class MockFileSandbox(SandboxBackend):
    """In-process mock filesystem sandbox.

    Tracks reads/writes so the behavior graph can derive reads/writes edges.
    Safe and deterministic — no real filesystem access outside the tmp dir.
    """

    def __init__(self, root: str = "/tmp/graphguard_sandbox"):
        self.root = root
        self._files: Dict[str, str] = {}
        self._read_log: List[str] = []
        self._write_log: List[str] = []

    def run(self, tool: str, arguments: Dict[str, Any]) -> SandboxResult:
        if tool == "file_read":
            path = arguments.get("path", "")
            self._read_log.append(path)
            if path in self._files:
                return SandboxResult(success=True, output=self._files[path])
            return SandboxResult(success=False, error=f"File not found: {path}")

        if tool == "file_write":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            self._files[path] = content
            self._write_log.append(path)
            return SandboxResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {path}",
                side_effects=[f"write:{path}"],
            )

        if tool == "file_list":
            return SandboxResult(success=True, output=list(self._files.keys()))

        return SandboxResult(success=False, error=f"Unknown file tool: {tool}")

    @property
    def reads(self) -> List[str]:
        return list(self._read_log)

    @property
    def writes(self) -> List[str]:
        return list(self._write_log)


class MockEmailSandbox(SandboxBackend):
    """Mock email sandbox for exfiltration / indirect-prompt-injection scenarios."""

    def __init__(self):
        self._sent: List[Dict[str, Any]] = []
        self._inbox: List[Dict[str, Any]] = []

    def run(self, tool: str, arguments: Dict[str, Any]) -> SandboxResult:
        if tool == "email_send":
            record = {
                "to": arguments.get("to"),
                "subject": arguments.get("subject"),
                "body": arguments.get("body"),
                "attachments": arguments.get("attachments", []),
            }
            self._sent.append(record)
            return SandboxResult(
                success=True,
                output=f"Email sent to {record['to']}",
                side_effects=[f"email_send:{record['to']}"],
            )

        if tool == "email_read":
            return SandboxResult(success=True, output=list(self._inbox))

        if tool == "email_receive":
            message = arguments.get("message", {})
            self._inbox.append(message)
            return SandboxResult(success=True, output="Message received")

        return SandboxResult(success=False, error=f"Unknown email tool: {tool}")

    @property
    def sent(self) -> List[Dict[str, Any]]:
        return list(self._sent)


class MockCodeSandbox(SandboxBackend):
    """Mock code-execution sandbox (read-only expression eval, safe subset)."""

    def run(self, tool: str, arguments: Dict[str, Any]) -> SandboxResult:
        if tool == "code_eval":
            expression = arguments.get("expression", "")
            try:
                # Extremely restricted: only literal eval of safe expressions.
                result = eval(expression, {"__builtins__": {}}, {})
                return SandboxResult(success=True, output=result)
            except Exception as exc:  # noqa: BLE001
                return SandboxResult(success=False, error=str(exc))
        return SandboxResult(success=False, error=f"Unknown code tool: {tool}")


class Sandbox:
    """Facade aggregating mock backends."""

    def __init__(self):
        self.backends: Dict[str, SandboxBackend] = {
            "file": MockFileSandbox(),
            "email": MockEmailSandbox(),
            "code": MockCodeSandbox(),
        }

    def run(self, tool: str, arguments: Dict[str, Any]) -> SandboxResult:
        # Route by tool prefix convention: "file_read", "email_send", etc.
        prefix = tool.split("_")[0]
        backend = self.backends.get(prefix)
        if backend is None:
            return SandboxResult(success=False, error=f"No backend for tool: {tool}")
        return backend.run(tool, arguments)

    @property
    def file_reads(self) -> List[str]:
        return self.backends["file"].reads  # type: ignore[attr-defined]

    @property
    def file_writes(self) -> List[str]:
        return self.backends["file"].writes  # type: ignore[attr-defined]

    @property
    def emails_sent(self) -> List[Dict[str, Any]]:
        return self.backends["email"].sent  # type: ignore[attr-defined]
