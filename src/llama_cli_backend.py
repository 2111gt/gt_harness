"""
Official llama.cpp CLI backend for GGUF inference.

Used when llama-cpp-python wheels crash with illegal instruction (0xc000001d)
on older CPUs. Downloads the official Windows CPU binary from ggml-org releases
and shells out to llama-cli for chat completions.

This still runs the **same on-disk Granite GGUF** — not a remote API and not
the rule-based offline draft.
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.request import urlretrieve

from .utils import MODELS_DIR, ensure_directories, run_hidden_subprocess, setup_logging

logger = setup_logging()

# Official CPU build (runtime-dispatched SIMD; more portable than some PyPI wheels)
LLAMA_CPP_RELEASE = os.environ.get("GT_LLAMA_CPP_RELEASE", "b9947")
LLAMA_CPP_ZIP_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_RELEASE}/llama-{LLAMA_CPP_RELEASE}-bin-win-cpu-x64.zip"
)
BIN_DIR = MODELS_DIR / "llama-cpp-bin"


class LlamaCliRunner:
    """
    Thin wrapper with a subset of the llama-cpp-python Llama API:
    create_chat_completion(...) and __call__(prompt, ...).
    """

    def __init__(
        self,
        model_path: str,
        cli_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        # Always absolute: generate() sets cwd to the binary folder so DLLs load,
        # and a relative models\*.gguf path would break under that cwd (TUI hang /
        # empty output).
        self.model_path = str(Path(model_path).resolve())
        self.cli_path = str(Path(cli_path).resolve())
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = "llama-cli"

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = _messages_to_prompt(messages)
        text = self._generate(
            prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    def __call__(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        text = self._generate(
            prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stop=stop,
        )
        return {"choices": [{"text": text}]}

    def _generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        # Prefer llama-completion for batch-style completion; fall back to llama-cli.
        cli = Path(self.cli_path)
        candidates = []
        for name in ("llama-completion.exe", "llama-completion", cli.name, "llama-cli.exe"):
            p = cli.parent / name
            if p.is_file() and p not in candidates:
                candidates.append(p)
        if cli.is_file() and cli not in candidates:
            candidates.insert(0, cli)

        last_err = ""
        n_threads = max(1, int(self.n_threads))
        # One preferred arg set; only fall back if the binary rejects flags.
        preferred_extra = [
            "-t",
            str(n_threads),
            "-tb",
            str(n_threads),
            "--temp",
            str(temperature),
            "-no-cnv",
            "--no-display-prompt",
        ]
        minimal_extra = [
            "-t",
            str(n_threads),
            "--temp",
            str(temperature),
            "-no-cnv",
        ]

        # Write prompt to a temp file (-f): avoids Windows cmdline limits and
        # interactive console behavior that hangs under Textual TUI.
        prompt_path: Optional[Path] = None
        try:
            fd, tmp_name = tempfile.mkstemp(prefix="gt_llama_prompt_", suffix=".txt")
            os.close(fd)
            prompt_path = Path(tmp_name)
            prompt_path.write_text(prompt, encoding="utf-8", errors="replace")

            env = os.environ.copy()
            env["LLAMA_LOG_COLORS"] = "0"
            env["LLAMA_LOG_VERBOSITY"] = env.get("LLAMA_LOG_VERBOSITY", "1")
            env.setdefault("TERM", "dumb")

            # At most a few attempts — each reloads ~5GB GGUF on CPU.
            attempts: List[Tuple[Path, List[str], List[str]]] = []
            for binary in candidates[:2]:
                attempts.append((binary, preferred_extra, ["-f", str(prompt_path)]))
                attempts.append((binary, minimal_extra, ["-f", str(prompt_path)]))
            # Last resort: short -p (may truncate long prompts)
            if candidates:
                attempts.append(
                    (candidates[0], minimal_extra, ["-p", prompt[:3500]])
                )

            seen_cmds: Set[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = set()
            for binary, extra, prompt_args in attempts:
                key = (str(binary), tuple(extra), tuple(prompt_args[:1]))
                if key in seen_cmds:
                    continue
                seen_cmds.add(key)
                cmd = [
                    str(binary),
                    "-m",
                    self.model_path,
                    "-n",
                    str(max(1, int(max_tokens))),
                    "-c",
                    str(self.n_ctx),
                    *extra,
                    *prompt_args,
                ]
                try:
                    # CREATE_NO_WINDOW: critical under Textual — without it
                    # llama-*.exe often hangs on the shared Windows console.
                    logger.info(
                        "llama-cli generate via %s (hidden console, n=%s, c=%s)",
                        binary.name,
                        max_tokens,
                        self.n_ctx,
                    )
                    proc = run_hidden_subprocess(
                        cmd,
                        timeout=600,
                        env=env,
                        cwd=str(binary.parent),
                    )
                except Exception as exc:  # noqa: BLE001
                    last_err = f"{binary.name}: {exc}"
                    logger.warning("llama-cli run failed: %s", exc)
                    continue

                raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
                text = _extract_generation(proc.stdout or "", prompt)
                if stop:
                    for s in stop:
                        if s and s in text:
                            text = text.split(s, 1)[0]
                text = text.strip()
                if not text:
                    text = _strip_cli_noise(proc.stdout or "")
                if text:
                    return text
                last_err = f"{binary.name} code={proc.returncode}: {raw[-500:]}"
                logger.warning(
                    "llama-cli empty output (code=%s): %s",
                    proc.returncode,
                    raw[-300:],
                )
                # Only keep trying if flags look unsupported; otherwise stop
                # reloading the multi-GB model over and over.
                low = raw.lower()
                if "unknown" in low or "unrecognized" in low or "invalid" in low:
                    continue
                break
        finally:
            if prompt_path is not None:
                try:
                    prompt_path.unlink(missing_ok=True)
                except Exception:
                    pass

        raise RuntimeError(f"llama.cpp CLI produced empty output: {last_err}")


def ensure_llama_cli_binary(force: bool = False) -> Optional[Path]:
    """
    Ensure official llama-cli.exe exists under models/llama-cpp-bin/.
    Downloads the win-cpu-x64 zip when missing.
    """
    ensure_directories()
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    existing = _find_cli(BIN_DIR)
    if existing and not force:
        return existing

    if (os.environ.get("GT_NO_DOWNLOAD") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        logger.warning("llama-cli missing and downloads disabled")
        return existing

    zip_path = BIN_DIR / f"llama-{LLAMA_CPP_RELEASE}-win-cpu-x64.zip"
    logger.info("Downloading official llama.cpp CPU binary: %s", LLAMA_CPP_ZIP_URL)
    try:
        urlretrieve(LLAMA_CPP_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BIN_DIR)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to download llama.cpp binary: %s", exc)
        return _find_cli(BIN_DIR)

    return _find_cli(BIN_DIR)


def try_load_llama_cli(
    model_path: Path,
    *,
    n_ctx: int = 2048,
    n_threads: int = 4,
    temperature: float = 0.2,
    max_tokens: int = 384,
) -> tuple[Optional[LlamaCliRunner], str]:
    """
    Build a LlamaCliRunner for the given GGUF, downloading CLI if needed.

    By default skips a full GGUF smoke generation (reloading 5GB is slow).
    Set GT_LLAMA_SMOKE=1 to force a short generate on bind.
    """
    cli = ensure_llama_cli_binary()
    if cli is None:
        return None, "llama-cli binary not available"

    if not Path(model_path).is_file():
        return None, f"GGUF missing: {model_path}"

    runner = LlamaCliRunner(
        model_path=str(model_path),
        cli_path=str(cli),
        n_ctx=n_ctx,
        n_threads=n_threads,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    smoke_flag = (os.environ.get("GT_LLAMA_SMOKE") or "").strip().lower()
    if smoke_flag in {"1", "true", "yes", "on"}:
        try:
            smoke = runner("OK", max_tokens=4, temperature=0.0)
            sample = (smoke.get("choices") or [{}])[0].get("text", "")
            return (
                runner,
                f"Loaded GGUF via llama-cli ({cli.name}): {Path(model_path).name} smoke={sample[:40]!r}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("llama-cli smoke test failed: %s", exc)
            return None, f"llama-cli failed to run GGUF: {exc}"

    # Fast path: binary + GGUF present (smoke deferred to first real generate)
    return (
        runner,
        f"Loaded GGUF via llama-cli ({cli.name}): {Path(model_path).name} (ready, smoke skipped)",
    )


def _find_cli(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    for name in (
        "llama-completion.exe",
        "llama-completion",
        "llama-cli.exe",
        "llama-cli",
        "main.exe",
    ):
        hits = list(root.rglob(name))
        if hits:
            return hits[0].resolve()
    return None


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").capitalize()
        content = (m.get("content") or "").strip()
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _extract_generation(stdout: str, prompt: str) -> str:
    """Pull model continuation from llama-cli stdout."""
    out = stdout or ""
    # If prompt is echoed, take text after last Assistant: / prompt
    for marker in ("Assistant:", prompt[-80:] if len(prompt) > 80 else prompt):
        if marker and marker in out:
            out = out.split(marker)[-1]
    return _strip_cli_noise(out)


def _strip_cli_noise(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        # Drop common llama.cpp log lines
        if re.match(r"^(main|llama_|ggml_|print_info|load_|system_info|sampling|generate:)", s, re.I):
            continue
        if s.startswith("llama_") or s.startswith("ggml_"):
            continue
        if "slot" in s.lower() and "processing" in s.lower():
            continue
        lines.append(line)
    return "\n".join(lines).strip()
