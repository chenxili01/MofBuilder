#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Literal, Optional


Step = Literal["planner", "executor", "reviewer"]
VALID_STEPS = {"planner", "executor", "reviewer"}

PHASE_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*Phase\s+(?P<number>\d+)\s*[-:–—]\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)
PHASE_NUMBER_RE = re.compile(r"\bPhase\s+(?P<number>\d+)\b")


@dataclass(frozen=True)
class Phase:
    number: int
    name: str


@dataclass(frozen=True)
class StatusSnapshot:
    phase_name: str
    phase_number: int
    checkpoint: str
    status: str
    next_step: str


# --------------------------------------------------
# Paths
# --------------------------------------------------

SCRIPT_PATH = pathlib.Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent
REPO_ROOT = ROOT.parent


def resolve_control_path(filename: str) -> pathlib.Path:
    repo_path = REPO_ROOT / filename
    if repo_path.exists():
        return repo_path
    return ROOT / filename


STATE_DIR = ROOT / "state"
STATE_FILE = STATE_DIR / "state.json"
LEGACY_STATE_FILE = ROOT / "workflow" / "state.json"

PLANNER_FILE = resolve_control_path("PLANNER.md")
EXECUTOR_FILE = resolve_control_path("EXECUTOR.md")
REVIEWER_FILE = resolve_control_path("REVIEWER.md")

STATUS_FILE = resolve_control_path("STATUS.md")
WORKLOG_FILE = resolve_control_path("WORKLOG.md")
REVIEW_FILE = resolve_control_path("REVIEW.md")
PLANS_FILE = resolve_control_path("PLANS.md")
CRASH_LOG_FILE = STATE_DIR / "crash.log"


# --------------------------------------------------
# Defaults
# --------------------------------------------------

DEFAULT_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.4")
DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("WORKFLOW_MAX_CONTEXT_CHARS", "24000"))

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "executor_can_proceed": {"type": "boolean"},
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["approved", "executor_can_proceed", "summary", "issues"],
    "additionalProperties": False,
}


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def today_iso() -> str:
    return datetime.now().date().isoformat()


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text_if_exists(path: pathlib.Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def write_text_atomic(path: pathlib.Path, text: str) -> None:
    ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.write(text)
        tmp_path = pathlib.Path(tmp.name)
    tmp_path.replace(path)


def write_json_atomic(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def append_text(path: pathlib.Path, text: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def append_markdown_section(path: pathlib.Path, label: str, text: str) -> None:
    stamp = utc_now_iso()
    block = f"\n\n## {label} ({stamp})\n\n{text.rstrip()}\n"
    append_text(path, block)


def tail_chars(text: str, limit: int, truncated_label: str) -> str:
    if limit <= 0:
        return f"[{truncated_label}: omitted]"
    if len(text) <= limit:
        return text
    return f"[{truncated_label}: showing last {limit} chars]\n{text[-limit:]}"


def section(name: str, text: str) -> str:
    return f"===== {name} =====\n{text.strip()}\n"


def load_phases(plans_path: pathlib.Path = PLANS_FILE) -> list[Phase]:
    text = read_text_if_exists(plans_path, "")
    phases = [
        Phase(number=int(match.group("number")), name=f"Phase {match.group('number')} - {match.group('title').strip()}")
        for match in PHASE_HEADING_RE.finditer(text)
    ]
    if not phases:
        raise RuntimeError(f"No phase headings were found in {plans_path.name}.")
    return phases


def find_phase_index(phases: list[Phase], phase_name: str) -> int:
    for idx, phase in enumerate(phases):
        if phase.name == phase_name:
            return idx

    phase_match = PHASE_NUMBER_RE.search(phase_name)
    if phase_match is None:
        raise ValueError(f"Phase was not found: {phase_name}")

    phase_number = int(phase_match.group("number"))
    for idx, phase in enumerate(phases):
        if phase.number == phase_number:
            return idx

    raise ValueError(f"Phase was not found: {phase_name}")


def _strip_status_value(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`") and len(value) >= 2:
        return value[1:-1]
    return value


def load_status_snapshot(status_path: pathlib.Path = STATUS_FILE) -> StatusSnapshot:
    text = read_text_if_exists(status_path, "")
    fields: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        fields[key.strip()] = _strip_status_value(value)

    phase_name = fields.get("Phase", "")
    phase_match = PHASE_NUMBER_RE.search(phase_name)
    if phase_match is None:
        raise RuntimeError(f"Could not parse current phase from {status_path.name}.")

    return StatusSnapshot(
        phase_name=phase_name,
        phase_number=int(phase_match.group("number")),
        checkpoint=fields.get("Checkpoint", ""),
        status=fields.get("Status", ""),
        next_step=fields.get("Next step", ""),
    )


# --------------------------------------------------
# Subprocess helpers
# --------------------------------------------------

def run_cmd(
    cmd: Iterable[str],
    *,
    cwd: pathlib.Path = ROOT,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=env,
    )
    if check and result.returncode != 0:
        command_str = " ".join(result.args)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command_str}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


# --------------------------------------------------
# Git helpers
# --------------------------------------------------

def git_diff(max_chars: int) -> str:
    diff = run_cmd(["git", "diff"], check=True).stdout
    return tail_chars(diff, max_chars, "git diff truncated")


def git_commit(tag: str) -> str:
    run_cmd(["git", "add", "-A"], check=True)

    msg = f"workflow checkpoint: {tag}"
    run_cmd(["git", "commit", "--allow-empty", "-m", msg], check=True)

    sha = run_cmd(["git", "rev-parse", "HEAD"], check=True).stdout.strip()
    if not sha:
        raise RuntimeError("git rev-parse HEAD returned an empty SHA")

    run_cmd(["git", "tag", "-f", tag], check=True)
    return sha


def git_phase_tag(phase_number: int, suffix: str) -> str:
    if suffix:
        return f"phase{phase_number}-{suffix}"
    return f"phase{phase_number}"


def verify_git_repo() -> None:
    run_cmd(["git", "rev-parse", "--is-inside-work-tree"], check=True)


# --------------------------------------------------
# State
# --------------------------------------------------

def normalize_step(value: Any) -> Step:
    if isinstance(value, str) and value in VALID_STEPS:
        return value  # type: ignore[return-value]
    return "planner"


def load_state() -> Dict[str, Step]:
    state_path = STATE_FILE if STATE_FILE.exists() else LEGACY_STATE_FILE
    if not state_path.exists():
        return {"step": "planner"}

    try:
        data = json.loads(read_text_if_exists(state_path, "{}"))
    except json.JSONDecodeError:
        print(f"Warning: malformed {state_path}; resetting to planner", file=sys.stderr)
        return {"step": "planner"}

    step = normalize_step(data.get("step"))
    if step != data.get("step"):
        print(f"Warning: invalid step in {state_path}; resetting to planner", file=sys.stderr)

    return {"step": step}


def save_state(step: Step) -> None:
    write_json_atomic(STATE_FILE, {"step": step})
    if LEGACY_STATE_FILE.exists():
        LEGACY_STATE_FILE.unlink(missing_ok=True)


def infer_step_from_status(snapshot: StatusSnapshot) -> Optional[Step]:
    next_step = snapshot.next_step.strip().lower()
    status = snapshot.status.strip().lower()

    if "reviewer" in next_step:
        return "reviewer"
    if "executor" in next_step or "implementation" in next_step:
        return "executor"
    if "planner" in next_step:
        return "planner"
    if status in {"pending", "contract generated"}:
        return "planner"
    return None


def resolve_resume_step(initial_step: Optional[Step]) -> Step:
    if initial_step is not None:
        return initial_step

    saved_step = load_state()["step"]
    status_snapshot = load_status_snapshot()
    status_step = infer_step_from_status(status_snapshot)

    if status_step is None:
        return saved_step

    if status_step != saved_step:
        print(
            "Workflow state disagrees with STATUS.md; "
            f"using {status_step} from STATUS.md instead of {saved_step}."
        )
        save_state(status_step)

    return status_step


def replace_status_field(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^- {re.escape(key)}:.*$")
    if not pattern.search(text):
        raise RuntimeError(f"Could not find '{key}' in {STATUS_FILE.name}.")
    return pattern.sub(f"- {key}: {value}", text, count=1)


def update_status_snapshot(
    *,
    phase_name: Optional[str] = None,
    checkpoint: Optional[str] = None,
    status: Optional[str] = None,
    next_step: Optional[str] = None,
    last_update: Optional[str] = None,
) -> None:
    text = read_text_if_exists(STATUS_FILE, "")

    updates = {
        "Phase": f"`{phase_name}`" if phase_name is not None else None,
        "Checkpoint": f"`{checkpoint}`" if checkpoint is not None else None,
        "Status": status,
        "Next step": next_step,
        "Last update": last_update,
    }

    for key, value in updates.items():
        if value is None:
            continue
        text = replace_status_field(text, key, value)

    write_text_atomic(STATUS_FILE, text)


def advance_status_to_next_phase(phases: list[Phase], current_snapshot: StatusSnapshot) -> Optional[Phase]:
    current_index = find_phase_index(phases, current_snapshot.phase_name)
    if current_index + 1 >= len(phases):
        return None

    next_phase = phases[current_index + 1]
    update_status_snapshot(
        phase_name=next_phase.name,
        checkpoint=f"P{next_phase.number}.0",
        status="pending",
        next_step="planner",
        last_update=today_iso(),
    )
    return next_phase


# --------------------------------------------------
# Context
# --------------------------------------------------

def read_context(max_context_chars: int) -> str:
    # Split the budget across sections to avoid unbounded growth.
    plans_budget = max(1500, max_context_chars // 5)
    status_budget = max(1500, max_context_chars // 5)
    worklog_budget = max(3000, max_context_chars // 4)
    review_budget = max(2000, max_context_chars // 6)
    diff_budget = max(3000, max_context_chars - (plans_budget + status_budget + worklog_budget + review_budget))

    parts = [
        section("PLANS.md", tail_chars(read_text_if_exists(PLANS_FILE, "[missing]"), plans_budget, "PLANS.md truncated")),
        section("STATUS.md", tail_chars(read_text_if_exists(STATUS_FILE, "[missing]"), status_budget, "STATUS.md truncated")),
        section("WORKLOG.md", tail_chars(read_text_if_exists(WORKLOG_FILE, "[missing]"), worklog_budget, "WORKLOG.md truncated")),
        section("REVIEW.md", tail_chars(read_text_if_exists(REVIEW_FILE, "[missing]"), review_budget, "REVIEW.md truncated")),
        section("GIT DIFF", git_diff(diff_budget)),
    ]
    return "\n\n".join(parts)


# --------------------------------------------------
# Prompt loading
# --------------------------------------------------

def load_prompt_file(path: pathlib.Path, title: str) -> str:
    text = read_text_if_exists(path, "").strip()
    if text:
        return text
    return (
        f"You are the {title} for this repository workflow.\n"
        f"The expected prompt file {path.name} is missing.\n"
        f"Proceed conservatively using the repository context only."
    )


# --------------------------------------------------
# Codex CLI
# --------------------------------------------------

def write_temp_schema() -> pathlib.Path:
    ensure_parent(STATE_DIR / "dummy")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(STATE_DIR),
        prefix=".review-schema.",
        suffix=".json",
    ) as tmp:
        json.dump(REVIEW_SCHEMA, tmp, indent=2)
        tmp.write("\n")
        return pathlib.Path(tmp.name)


def build_codex_prompt(role_prompt: str, context: str, extra: str = "") -> str:
    extra_block = f"\n\nADDITIONAL INSTRUCTIONS\n{extra.strip()}\n" if extra.strip() else ""
    return (
        f"{role_prompt.strip()}\n\n"
        f"REPOSITORY CONTEXT\n{context.strip()}"
        f"{extra_block}"
    )


def run_codex_exec(
    prompt: str,
    *,
    model: str,
    allow_edits: bool,
    use_search: bool,
    schema_path: Optional[pathlib.Path] = None,
) -> str:
    if not command_exists("codex"):
        raise RuntimeError(
            "Codex CLI was not found on PATH. Install it first, then sign in with ChatGPT or configure auth."
        )

    cmd = ["codex"]

    if allow_edits:
        cmd.append("exec")
    else:
        # `codex exec` no longer accepts `--ask-for-approval` after the
        # subcommand, so keep the approval policy at the top level.
        cmd.extend(["--ask-for-approval", "never", "exec"])

    cmd.extend(["--model", model])

    if allow_edits:
        cmd.extend(["--full-auto"])
    else:
        cmd.extend(["--sandbox", "read-only"])

    if use_search:
        cmd.append("--search")

    if schema_path is not None:
        cmd.extend(["--output-schema", str(schema_path)])

    cmd.append(prompt)

    result = run_cmd(cmd, check=True)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")

    return result.stdout.strip()


# --------------------------------------------------
# Review parsing
# --------------------------------------------------

def parse_review_result(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Reviewer returned invalid JSON: {e}") from e

    required = {"approved", "executor_can_proceed", "summary", "issues"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Reviewer JSON missing required keys: {sorted(missing)}")

    if not isinstance(data["approved"], bool):
        raise ValueError("'approved' must be a boolean")
    if not isinstance(data["executor_can_proceed"], bool):
        raise ValueError("'executor_can_proceed' must be a boolean")
    if not isinstance(data["summary"], str):
        raise ValueError("'summary' must be a string")
    if not isinstance(data["issues"], list) or not all(isinstance(x, str) for x in data["issues"]):
        raise ValueError("'issues' must be a list of strings")

    return data


# --------------------------------------------------
# Agents
# --------------------------------------------------

def run_planner(model: str, max_context_chars: int) -> str:
    prompt_text = load_prompt_file(PLANNER_FILE, "planner")
    ctx = read_context(max_context_chars)
    prompt = build_codex_prompt(
        prompt_text,
        ctx,
        extra=(
            "Write a concise planning update. Update repository files if needed. "
            "Be explicit about the next executor step."
        ),
    )
    return run_codex_exec(prompt, model=model, allow_edits=True, use_search=False)


def run_executor(model: str, max_context_chars: int) -> str:
    prompt_text = load_prompt_file(EXECUTOR_FILE, "executor")
    ctx = read_context(max_context_chars)
    prompt = build_codex_prompt(
        prompt_text,
        ctx,
        extra=(
            "Make the required repository changes, run relevant commands/tests if appropriate, "
            "and summarize exactly what changed."
        ),
    )
    return run_codex_exec(prompt, model=model, allow_edits=True, use_search=False)


def run_reviewer(model: str, max_context_chars: int) -> Dict[str, Any]:
    prompt_text = load_prompt_file(REVIEWER_FILE, "reviewer")
    ctx = read_context(max_context_chars)
    schema_path = write_temp_schema()

    try:
        prompt = build_codex_prompt(
            prompt_text,
            ctx,
            extra=(
                "Review the current repository state and diff. "
                "Return only a JSON object that matches the provided schema. "
                "Set approved=true only if the phase is ready. "
                "Set executor_can_proceed=true only if executor should continue next."
            ),
        )
        raw = run_codex_exec(
            prompt,
            model=model,
            allow_edits=False,
            use_search=False,
            schema_path=schema_path,
        )
    finally:
        try:
            schema_path.unlink(missing_ok=True)
        except Exception:
            pass

    append_markdown_section(REVIEW_FILE, "reviewer-raw", raw)

    try:
        parsed = parse_review_result(raw)
    except ValueError as e:
        debug_payload = {
            "approved": False,
            "executor_can_proceed": False,
            "summary": f"Reviewer output invalid: {e}",
            "issues": [str(e)],
            "raw_output": raw,
        }
        append_markdown_section(REVIEW_FILE, "reviewer-parse-error", json.dumps(debug_payload, indent=2))
        return debug_payload

    append_markdown_section(REVIEW_FILE, "reviewer-parsed", json.dumps(parsed, indent=2))
    return parsed


# --------------------------------------------------
# Workflow
# --------------------------------------------------

def remediation_step(model: str, max_context_chars: int) -> None:
    print("Review failed -> remediation planning")
    run_planner(model=model, max_context_chars=max_context_chars)


def workflow(
    *,
    initial_step: Optional[Step],
    model: str,
    max_context_chars: int,
    no_git: bool,
) -> None:
    step = resolve_resume_step(initial_step)
    phases = load_phases()
    final_phase_number = phases[-1].number
    last_approved_phase_number: Optional[int] = None

    if not no_git:
        verify_git_repo()

    try:
        while True:
            status_snapshot = load_status_snapshot()
            print(f"Resuming at: {step} ({status_snapshot.phase_name} / {status_snapshot.checkpoint})")

            if step == "planner":
                print("Running planner...")
                run_planner(model=model, max_context_chars=max_context_chars)
                planner_snapshot = load_status_snapshot()
                if (
                    last_approved_phase_number is not None
                    and planner_snapshot.phase_number == last_approved_phase_number
                ):
                    print(
                        "Planner did not advance beyond the previously approved phase; "
                        "stopping to avoid repeating the same phase."
                    )
                    save_state("planner")
                    return
                save_state("executor")
                step = "executor"
                continue

            if step == "executor":
                print("Running executor...")
                executor_snapshot = load_status_snapshot()
                run_executor(model=model, max_context_chars=max_context_chars)
                if not no_git:
                    tag = git_phase_tag(executor_snapshot.phase_number, "executor-checkpoint")
                    sha = git_commit(tag)
                    print(f"Executor checkpoint ({tag}): {sha}")
                save_state("reviewer")
                step = "reviewer"
                continue

            print("Running reviewer...")
            reviewer_snapshot = load_status_snapshot()
            review = run_reviewer(model=model, max_context_chars=max_context_chars)

            approved = bool(review.get("approved"))
            executor_can_proceed = bool(review.get("executor_can_proceed"))

            print(f"Review summary: {review.get('summary', '')}")

            if approved:
                last_approved_phase_number = reviewer_snapshot.phase_number
                if not no_git:
                    tag = git_phase_tag(reviewer_snapshot.phase_number, "checkpoint")
                    sha = git_commit(tag)
                    print(f"Phase checkpoint ({tag}): {sha}")
                save_state("planner")
                if reviewer_snapshot.phase_number >= final_phase_number:
                    print("Final phase approved; workflow is complete.")
                    return
                next_phase = advance_status_to_next_phase(phases, reviewer_snapshot)
                if next_phase is None:
                    print("No further phase was found after approval; workflow is complete.")
                    return
                print(f"Advancing to: {next_phase.name} / P{next_phase.number}.0")
                step = "planner"
                continue

            if not no_git:
                tag = git_phase_tag(reviewer_snapshot.phase_number, "review-failure")
                sha = git_commit(tag)
                print(f"Review failure checkpoint ({tag}): {sha}")

            remediation_step(model=model, max_context_chars=max_context_chars)

            next_step: Step = "executor" if executor_can_proceed else "planner"
            save_state(next_step)
            print(f"Next step: {next_step}")
            return

    except Exception as exc:
        print(f"Workflow failure: {exc}", file=sys.stderr)
        tb = traceback.format_exc()
        append_markdown_section(CRASH_LOG_FILE, "workflow-crash", tb)

        if not no_git:
            try:
                crash_phase = load_status_snapshot().phase_number
                git_commit(git_phase_tag(crash_phase, "workflow-crash"))
            except Exception as git_exc:
                print(f"Crash checkpoint failed: {git_exc}", file=sys.stderr)

        raise


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planner/executor/reviewer workflow with Codex CLI.")
    parser.add_argument(
        "--step",
        choices=sorted(VALID_STEPS),
        help="Force the starting step instead of using the saved workflow state.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Codex model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip git commit/tag actions.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Maximum approximate context size. Default: {DEFAULT_MAX_CONTEXT_CHARS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workflow(
        initial_step=args.step,
        model=args.model,
        max_context_chars=max(4000, args.max_context_chars),
        no_git=bool(args.no_git),
    )


if __name__ == "__main__":
    main()
