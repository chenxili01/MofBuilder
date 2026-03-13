#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from typing import Any, Iterable, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - only unavailable on non-POSIX systems
    fcntl = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None


VALID_STEPS = ("planner", "executor", "reviewer")
DEFAULT_STEP = "planner"
DEFAULT_MODEL = (
    os.environ.get("OPENAI_RESPONSES_MODEL")
    or os.environ.get("OPENAI_MODEL")
    or "gpt-5.4"
)
DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("RUN_PHASES_MAX_CONTEXT_CHARS", "120000"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("RUN_PHASES_REQUEST_TIMEOUT_SECONDS", "180"))
COMMAND_TIMEOUT_SECONDS = float(os.environ.get("RUN_PHASES_COMMAND_TIMEOUT_SECONDS", "60"))
AGENT_RETRIES = int(os.environ.get("RUN_PHASES_AGENT_RETRIES", "2"))
COMMAND_RETRIES = int(os.environ.get("RUN_PHASES_COMMAND_RETRIES", "1"))
REVIEW_PARSE_RETRIES = int(os.environ.get("RUN_PHASES_REVIEW_PARSE_RETRIES", "1"))

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW_DIR = ROOT / "workflow"
STATE_DIR = WORKFLOW_DIR / "state"
STATE_FILE = STATE_DIR / "state.json"
LEGACY_STATE_FILE = WORKFLOW_DIR / "state.json"

PROMPT_FILES = {
    "planner": ROOT / "PLANNER.md",
    "executor": ROOT / "EXECUTOR.md",
    "reviewer": ROOT / "REVIEWER.md",
}

PLANS = ROOT / "PLANS.md"
STATUS = ROOT / "STATUS.md"
WORKLOG = ROOT / "WORKLOG.md"
REVIEW = ROOT / "REVIEW.md"
AGENTS = ROOT / "AGENTS.md"
ARCHITECTURE = ROOT / "ARCHITECTURE.md"
CODEX_CONTEXT = ROOT / "CODEX_CONTEXT.md"

_CLIENT: OpenAI | None = None

REVIEW_JSON_INSTRUCTIONS = """
Return only one JSON object. Do not include markdown fences or commentary.
Use exactly these keys:
{
  "review_decision": "APPROVED" | "FAILED",
  "phase": "<phase name>",
  "checkpoint": "<checkpoint name>",
  "can_executor_proceed": true | false,
  "blocking_findings": ["..."],
  "required_fixes": ["..."],
  "scope_violations": ["..."],
  "architecture_or_compatibility_risks": ["..."],
  "required_tests_before_approval": ["..."],
  "required_log_status_corrections": ["..."]
}
Rules:
- Every list field must be a JSON array of strings. Use [] when there is nothing to report.
- "review_decision" and "can_executor_proceed" must agree: APPROVED => true, FAILED => false.
- Keep string values concise and specific.
""".strip()


class WorkflowError(RuntimeError):
    """Workflow failure with a user-actionable message."""


class GitCommandError(WorkflowError):
    """Raised when a git command fails."""


class ReviewerOutputError(WorkflowError):
    """Raised when reviewer output is missing or malformed."""


@dataclass(frozen=True)
class WorkflowConfig:
    step_override: str | None
    model: str
    use_git: bool
    max_context_chars: int


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class ContextItem:
    label: str
    content: str
    strategy: str
    cap: int


@dataclass(frozen=True)
class ReviewResult:
    review_decision: str
    phase: str
    checkpoint: str
    can_executor_proceed: bool
    blocking_findings: list[str]
    required_fixes: list[str]
    scope_violations: list[str]
    architecture_or_compatibility_risks: list[str]
    required_tests_before_approval: list[str]
    required_log_status_corrections: list[str]
    raw_output: str

    @property
    def approved(self) -> bool:
        return self.review_decision == "APPROVED" and self.can_executor_proceed

    def to_json(self) -> str:
        payload = {
            "review_decision": self.review_decision,
            "phase": self.phase,
            "checkpoint": self.checkpoint,
            "can_executor_proceed": self.can_executor_proceed,
            "blocking_findings": self.blocking_findings,
            "required_fixes": self.required_fixes,
            "scope_violations": self.scope_violations,
            "architecture_or_compatibility_risks": self.architecture_or_compatibility_risks,
            "required_tests_before_approval": self.required_tests_before_approval,
            "required_log_status_corrections": self.required_log_status_corrections,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stderr(message: str) -> None:
    print(message, file=sys.stderr)


def validate_step(step: str) -> str:
    if step not in VALID_STEPS:
        raise WorkflowError(
            f"Invalid step {step!r}. Expected one of: {', '.join(VALID_STEPS)}."
        )
    return step


def read_text(path: pathlib.Path, *, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def read_required_text(path: pathlib.Path, description: str) -> str:
    if not path.exists():
        raise WorkflowError(f"Required {description} is missing: {path}")
    return path.read_text(encoding="utf-8")


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = pathlib.Path(handle.name)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
        raise


@contextlib.contextmanager
def file_lock(lock_path: pathlib.Path) -> Iterable[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_text_atomic(path: pathlib.Path, text: str) -> None:
    payload = text.strip()
    if not payload:
        return

    with file_lock(path.with_name(f"{path.name}.lock")):
        existing = read_text(path, default="").rstrip()
        chunks = [chunk for chunk in (existing, payload) if chunk]
        final_text = "\n\n".join(chunks) + "\n"
        atomic_write_text(path, final_text)


def backup_corrupt_state(path: pathlib.Path, reason: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.corrupt-{timestamp}{path.suffix or '.json'}")
    try:
        path.replace(backup)
        stderr(f"Recovered from malformed workflow state ({reason}); backed up to {backup}.")
    except OSError as exc:
        stderr(
            f"Recovered from malformed workflow state ({reason}) but could not back it up: {exc}"
        )


def current_state_file() -> pathlib.Path:
    if STATE_FILE.exists():
        return STATE_FILE
    if LEGACY_STATE_FILE.exists():
        return LEGACY_STATE_FILE
    return STATE_FILE


def load_state(step_override: str | None) -> dict[str, str]:
    if step_override is not None:
        return {"step": validate_step(step_override)}

    path = current_state_file()
    if not path.exists():
        return {"step": DEFAULT_STEP}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        backup_corrupt_state(path, "invalid JSON")
        return {"step": DEFAULT_STEP}

    if not isinstance(data, dict):
        backup_corrupt_state(path, "state root was not a JSON object")
        return {"step": DEFAULT_STEP}

    step = data.get("step")
    if not isinstance(step, str) or step not in VALID_STEPS:
        backup_corrupt_state(path, "missing or invalid step field")
        return {"step": DEFAULT_STEP}

    return {"step": step}


def save_state(state: dict[str, str]) -> None:
    payload = {"step": validate_step(state["step"])}
    atomic_write_text(STATE_FILE, json.dumps(payload, indent=2) + "\n")


def render_missing_file(path: pathlib.Path) -> str:
    return f"[missing file: {path.name}]"


def truncate_text(text: str, limit: int, strategy: str) -> str:
    if limit <= 0 or len(text) <= limit:
        return text

    if limit < 64:
        return text[:limit]

    removed = len(text) - limit
    if strategy == "tail":
        body = text[-(limit - 40) :]
        return f"... [truncated {removed} leading chars] ...\n{body}"
    if strategy == "balanced":
        note = f"\n... [truncated {removed} middle chars] ...\n"
        remaining = limit - len(note)
        head = remaining // 2
        tail = remaining - head
        return text[:head] + note + text[-tail:]

    body = text[: limit - 41]
    return f"{body}\n... [truncated {removed} trailing chars] ..."


def scale_caps(items: Sequence[ContextItem], total_limit: int) -> list[int]:
    if not items:
        return []

    caps = [max(1, item.cap) for item in items]
    cap_sum = sum(caps)
    if cap_sum <= total_limit:
        return caps

    scaled = [max(1, int(cap * total_limit / cap_sum)) for cap in caps]
    while sum(scaled) > total_limit:
        index = max(range(len(scaled)), key=scaled.__getitem__)
        if scaled[index] <= 1:
            break
        scaled[index] -= 1
    return scaled


def build_context(config: WorkflowConfig, step: str) -> str:
    items: list[ContextItem] = [
        ContextItem("PLANS.md", read_text(PLANS, default=render_missing_file(PLANS)), "head", 16000),
        ContextItem("AGENTS.md", read_text(AGENTS, default=render_missing_file(AGENTS)), "head", 24000),
        ContextItem(
            "ARCHITECTURE.md",
            read_text(ARCHITECTURE, default=render_missing_file(ARCHITECTURE)),
            "head",
            12000,
        ),
        ContextItem(
            "CODEX_CONTEXT.md",
            read_text(CODEX_CONTEXT, default=render_missing_file(CODEX_CONTEXT)),
            "head",
            12000,
        ),
        ContextItem("STATUS.md", read_text(STATUS, default=render_missing_file(STATUS)), "tail", 5000),
        ContextItem("WORKLOG.md", read_text(WORKLOG, default=render_missing_file(WORKLOG)), "tail", 30000),
    ]

    if step == "planner":
        items.append(
            ContextItem("REVIEW.md", read_text(REVIEW, default=render_missing_file(REVIEW)), "tail", 18000)
        )

    if config.use_git:
        diff_text = git_diff()
        if diff_text.strip():
            items.append(ContextItem("GIT DIFF", diff_text, "balanced", 18000))
    else:
        items.append(ContextItem("GIT DIFF", "[git disabled via --no-git]", "head", 200))

    separator_overhead = max(0, (len(items) - 1) * 2)
    label_overhead = sum(len(item.label) + 1 for item in items)
    content_budget = max(1, config.max_context_chars - separator_overhead - label_overhead)
    caps = scale_caps(items, content_budget)
    rendered = []
    for item, cap in zip(items, caps, strict=True):
        rendered.append(f"{item.label}\n{truncate_text(item.content, cap, item.strategy)}")
    return "\n\n".join(rendered)


def get_client() -> OpenAI:
    global _CLIENT
    if OpenAI is None:
        raise WorkflowError("The openai package is not installed; cannot run workflow agents.")
    if _CLIENT is None:
        _CLIENT = OpenAI(max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS)
    return _CLIENT


def run_command(
    cmd: Sequence[str],
    *,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
    retries: int = COMMAND_RETRIES,
    check: bool = True,
) -> CommandResult:
    last_error: Exception | None = None
    cmd_list = [str(part) for part in cmd]

    for attempt in range(retries + 1):
        try:
            completed = subprocess.run(
                cmd_list,
                text=True,
                capture_output=True,
                cwd=ROOT,
                encoding="utf-8",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = WorkflowError(
                f"Command timed out after {timeout:.0f}s: {' '.join(cmd_list)}"
            )
        except OSError as exc:
            last_error = WorkflowError(f"Failed to run command {' '.join(cmd_list)}: {exc}")
        else:
            result = CommandResult(
                args=tuple(cmd_list),
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )
            if not check or completed.returncode == 0:
                return result
            last_error = WorkflowError(
                "Command failed with exit code "
                f"{completed.returncode}: {' '.join(cmd_list)}\n"
                f"stdout:\n{completed.stdout or '[empty]'}\n"
                f"stderr:\n{completed.stderr or '[empty]'}"
            )

        if attempt < retries:
            time.sleep(2**attempt)

    assert last_error is not None
    raise last_error


def git_diff() -> str:
    return run_command(["git", "diff"]).stdout


def git_commit(tag: str) -> str:
    try:
        run_command(["git", "add", "-A"])
        message = f"workflow checkpoint: {tag}"
        run_command(["git", "commit", "--allow-empty", "-m", message])
        sha = run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
        run_command(["git", "tag", "-f", tag])
        return sha
    except WorkflowError as exc:
        raise GitCommandError(str(exc)) from exc


def best_effort_git_commit(tag: str, *, enabled: bool) -> None:
    if not enabled:
        return
    try:
        git_commit(tag)
    except Exception as exc:  # pragma: no cover - best-effort crash path
        stderr(f"Best-effort git checkpoint failed ({tag}): {exc}")


def run_agent(prompt: str, *, model: str) -> str:
    client = get_client()
    last_error: Exception | None = None

    for attempt in range(AGENT_RETRIES + 1):
        try:
            response = client.responses.create(model=model, input=prompt)
            output_text = getattr(response, "output_text", "")
        except Exception as exc:
            last_error = exc
        else:
            if output_text and output_text.strip():
                return output_text.strip()
            last_error = WorkflowError("OpenAI response did not contain any text output.")

        if attempt < AGENT_RETRIES:
            time.sleep(2**attempt)

    raise WorkflowError(f"Agent request failed after retries: {last_error}") from last_error


def normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        raise ReviewerOutputError(f"Reviewer JSON field {field_name!r} is missing.")
    if isinstance(value, str):
        item = value.strip()
        if not item or item.lower() == "none":
            return []
        return [item]
    if isinstance(value, list):
        normalized = []
        for item in value:
            text = str(item).strip()
            if not text or text.lower() == "none":
                continue
            normalized.append(text)
        return normalized
    raise ReviewerOutputError(f"Reviewer JSON field {field_name!r} must be a string or list.")


def normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    raise ReviewerOutputError(f"Reviewer JSON field {field_name!r} must be a boolean.")


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ReviewerOutputError("Reviewer returned empty output.")

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(candidate[index:])
        except JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        trailing = candidate[index + end :].strip()
        if trailing and not trailing.startswith("```"):
            continue
        return value

    raise ReviewerOutputError("Reviewer output did not contain a valid JSON object.")


def parse_review_result(raw_output: str) -> ReviewResult:
    payload = extract_json_object(raw_output)

    decision = str(payload.get("review_decision", "")).strip().upper()
    if decision not in {"APPROVED", "FAILED"}:
        raise ReviewerOutputError("Reviewer JSON field 'review_decision' must be APPROVED or FAILED.")

    phase = str(payload.get("phase", "")).strip()
    checkpoint = str(payload.get("checkpoint", "")).strip()
    if not phase or not checkpoint:
        raise ReviewerOutputError("Reviewer JSON must include non-empty 'phase' and 'checkpoint'.")

    can_executor_proceed = normalize_bool(
        payload.get("can_executor_proceed"), "can_executor_proceed"
    )
    if (decision == "APPROVED") != can_executor_proceed:
        raise ReviewerOutputError(
            "Reviewer JSON fields 'review_decision' and 'can_executor_proceed' disagree."
        )

    return ReviewResult(
        review_decision=decision,
        phase=phase,
        checkpoint=checkpoint,
        can_executor_proceed=can_executor_proceed,
        blocking_findings=normalize_string_list(payload.get("blocking_findings"), "blocking_findings"),
        required_fixes=normalize_string_list(payload.get("required_fixes"), "required_fixes"),
        scope_violations=normalize_string_list(payload.get("scope_violations"), "scope_violations"),
        architecture_or_compatibility_risks=normalize_string_list(
            payload.get("architecture_or_compatibility_risks"),
            "architecture_or_compatibility_risks",
        ),
        required_tests_before_approval=normalize_string_list(
            payload.get("required_tests_before_approval"),
            "required_tests_before_approval",
        ),
        required_log_status_corrections=normalize_string_list(
            payload.get("required_log_status_corrections"),
            "required_log_status_corrections",
        ),
        raw_output=raw_output.strip(),
    )


def format_review_block(items: Sequence[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def format_review_entry(review: ReviewResult) -> str:
    return (
        f"## Review {utc_timestamp()}\n\n"
        "```json\n"
        f"{review.to_json()}\n"
        "```\n\n"
        f"Review decision: {review.review_decision}\n"
        f"Phase: {review.phase}\n"
        f"Checkpoint: {review.checkpoint}\n"
        f"Can executor proceed?: {'yes' if review.can_executor_proceed else 'no'}\n\n"
        "Blocking findings:\n"
        f"{format_review_block(review.blocking_findings)}\n\n"
        "Required fixes:\n"
        f"{format_review_block(review.required_fixes)}\n\n"
        "Scope violations:\n"
        f"{format_review_block(review.scope_violations)}\n\n"
        "Architecture / compatibility risks:\n"
        f"{format_review_block(review.architecture_or_compatibility_risks)}\n\n"
        "Required tests before approval:\n"
        f"{format_review_block(review.required_tests_before_approval)}\n\n"
        "Required log/status corrections:\n"
        f"{format_review_block(review.required_log_status_corrections)}"
    )


def build_prompt(step: str, config: WorkflowConfig) -> str:
    prompt_text = read_required_text(PROMPT_FILES[step], f"{step} prompt")
    sections = [prompt_text.strip()]
    if step == "reviewer":
        sections.append(REVIEW_JSON_INSTRUCTIONS)
    sections.append("REPOSITORY CONTEXT")
    sections.append(build_context(config, step))
    return "\n\n".join(section for section in sections if section)


def run_planner(config: WorkflowConfig) -> str:
    output = run_agent(build_prompt("planner", config), model=config.model)
    append_text_atomic(WORKLOG, output)
    return output


def run_executor(config: WorkflowConfig) -> str:
    output = run_agent(build_prompt("executor", config), model=config.model)
    append_text_atomic(WORKLOG, output)
    return output


def run_reviewer(config: WorkflowConfig) -> ReviewResult:
    prompt = build_prompt("reviewer", config)
    last_error: ReviewerOutputError | None = None

    for attempt in range(REVIEW_PARSE_RETRIES + 1):
        raw_output = run_agent(prompt, model=config.model)
        try:
            review = parse_review_result(raw_output)
        except ReviewerOutputError as exc:
            last_error = exc
            if attempt < REVIEW_PARSE_RETRIES:
                prompt = (
                    f"{prompt}\n\n"
                    "Your previous response was invalid. "
                    f"Return only valid JSON that matches the schema. Error: {exc}"
                )
                continue
            raise
        append_text_atomic(REVIEW, format_review_entry(review))
        return review

    assert last_error is not None
    raise last_error


def workflow(config: WorkflowConfig) -> None:
    state = load_state(config.step_override)
    step = validate_step(state["step"])
    print("Resuming at:", step)

    try:
        if step == "planner":
            run_planner(config)
            save_state({"step": "executor"})
            step = "executor"

        if step == "executor":
            run_executor(config)
            if config.use_git:
                git_commit("executor-checkpoint")
            save_state({"step": "reviewer"})
            step = "reviewer"

        if step == "reviewer":
            review = run_reviewer(config)
            if review.approved:
                if config.use_git:
                    sha = git_commit("phase-approved")
                    print("Phase approved:", sha)
                else:
                    print("Phase approved.")
                save_state({"step": "planner"})
            else:
                print("Review failed -> remediation")
                if config.use_git:
                    git_commit("review-failure")
                run_planner(config)
                save_state({"step": "executor"})

    except Exception:
        best_effort_git_commit("workflow-crash", enabled=config.use_git)
        raise


def parse_args(argv: Sequence[str] | None = None) -> WorkflowConfig:
    parser = argparse.ArgumentParser(description="Run the planner/executor/reviewer workflow.")
    parser.add_argument(
        "--step",
        choices=VALID_STEPS,
        help="Override the saved workflow state and start from the given step.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use for all agents. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Disable git diff/context and skip workflow checkpoint commits/tags.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Maximum total repository-context characters to include. Default: {DEFAULT_MAX_CONTEXT_CHARS}",
    )
    args = parser.parse_args(argv)

    if args.max_context_chars <= 0:
        raise WorkflowError("--max-context-chars must be a positive integer.")

    return WorkflowConfig(
        step_override=args.step,
        model=args.model,
        use_git=not args.no_git,
        max_context_chars=args.max_context_chars,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        workflow(config)
    except WorkflowError as exc:
        stderr(f"Workflow failure: {exc}")
        return 1
    except KeyboardInterrupt:
        stderr("Workflow interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
