"""Provider-neutral LLM calls, driven entirely by config/agents.yaml.

A provider is a command-line tool described in config, so moving to another one means editing that file. Nothing
outside this module knows which provider is in use.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mdk_trading_oracle.core.config import get_settings


class ProviderError(RuntimeError):
    """The provider could not produce a usable result."""


@dataclass
class LLMResult:
    data: dict[str, Any]
    provider: str
    model: str
    duration_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)


def load_agents_config() -> dict[str, Any]:
    return get_settings().load_yaml("agents.yaml")


def _dig(obj: Any, path: str | None) -> Any:
    if not path:
        return None
    for part in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def resolve_binary(spec: dict[str, Any]) -> str:
    found = shutil.which(spec["binary"])
    if found:
        return found
    for pattern in spec.get("binary_fallback_globs", []):
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            return matches[-1]
    raise ProviderError(f"Binary {spec['binary']!r} not found on PATH or in configured fallback locations")


def parse_json_text(text: str) -> dict[str, Any]:
    """Parse model text as JSON, tolerating a surrounding markdown fence from providers without schema enforcement."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ProviderError(f"Provider returned text that is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ProviderError("Provider returned JSON that is not an object")
    return parsed


class CliProvider:
    def __init__(self, name: str, spec: dict[str, Any]):
        self.name = name
        self.spec = spec

    def model_for_tier(self, tier: str) -> str:
        return str(self.spec["models"][tier])

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        for var in self.spec.get("forbid_env", []):
            if env.get(var):
                raise ProviderError(
                    f"{var} is set, which would bill an API account instead of the subscription. Unset it to continue."
                )
        for var in self.spec.get("drop_env", []):
            env.pop(var, None)
        return env

    def complete_json(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any], model: str, timeout_seconds: int
    ) -> LLMResult:
        env = self._build_env()
        values = {"{model}": model, "{schema_json}": json.dumps(schema, separators=(",", ":")), "{system_prompt}": system_prompt}
        args = [values.get(a, a) for a in self.spec["args"]]
        command = [resolve_binary(self.spec), *args]
        start = time.monotonic()
        # An empty working directory stops the CLI loading this repo's project instructions on every call.
        with tempfile.TemporaryDirectory() as workdir:
            try:
                proc = subprocess.run(
                    command, input=user_prompt, capture_output=True, text=True, timeout=timeout_seconds, cwd=workdir, env=env
                )
            except subprocess.TimeoutExpired as e:
                raise ProviderError(f"{self.name} timed out after {timeout_seconds}s") from e
        duration = time.monotonic() - start
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[:500]
            raise ProviderError(f"{self.name} exited with {proc.returncode}: {detail}")
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise ProviderError(f"{self.name} returned output that is not JSON: {proc.stdout[:300]}") from e
        if _dig(envelope, self.spec.get("error_flag_path")):
            raise ProviderError(f"{self.name} reported an error: {str(_dig(envelope, self.spec.get('text_output_path')))[:500]}")
        data = _dig(envelope, self.spec.get("structured_output_path"))
        if not isinstance(data, dict):
            text = _dig(envelope, self.spec.get("text_output_path"))
            if not isinstance(text, str):
                raise ProviderError(f"{self.name} returned no structured output or text")
            data = parse_json_text(text)
        usage = {k: _dig(envelope, p) for k, p in self.spec.get("usage_paths", {}).items()}
        return LLMResult(data=data, provider=self.name, model=model, duration_seconds=duration, usage=usage)


def get_provider(name: str | None = None) -> CliProvider:
    cfg = load_agents_config()
    provider_name = name or cfg["active_provider"]
    if provider_name not in cfg["providers"]:
        raise ProviderError(f"Unknown provider {provider_name!r}; configured: {sorted(cfg['providers'])}")
    return CliProvider(provider_name, cfg["providers"][provider_name])


def read_prompt(relative_path: str) -> str:
    return Path(get_settings().project_root, relative_path).read_text(encoding="utf-8")
