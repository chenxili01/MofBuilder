#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

Step = Literal["planner", "executor"]
VALID_STEPS = {"planner", "executor"}

PHASE_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*Phase\s+(?P<number>\d+)\s*[—\-:–]\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)
STATUS_FIELD_RE = re.compile(r"(?m)^- (?P<key>[^:]+): (?P<value>.*)$")


@dataclass(frozen=True)
class Phase:
    number: int
    name: str


@dataclass(frozen=True)
class StatusSnapshot:
    phase_raw: str
    phase_number: int
    checkpoint: str
    status: str
    next_step: str


SCRIPT_PATH = pathlib.Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent
REPO_ROOT = ROOT.parent if (ROOT.parent / ".git").exists() else ROOT


def resolve_control_path(filename: str) -> pathlib.Path:
    repo_candidate = REPO_ROOT / filename
    if repo_candidate.exists():
        return repo_candidate
    return ROOT / filename


STATE_DIR = ROOT / "state"
STATE_FILE = STATE_DIR / "state.json"
CRASH_LOG_FILE = STATE_DIR / "crash.log"

STATUS_FILE = resolve_control_path("STATUS.md")
WORKLOG_FILE = resolve_control_path("WORKLOG.md")
PLAN_FILE = resolve_control_path("PLAN.md")
AGENTS_FILE = resolve_control_path("AGENTS.md")
CHECKLIST_FILE = resolve_control_path("CHECKLIST.md")
PHASE_SPEC_FILE = resolve_control_path("PHASE_SPEC.md")
ARCHITECTURE_FILE = resolve_control_path("ARCHITECTURE.md")
ARCHITECTURE_DECISIONS_FILE = resolve_control_path("ARCHITECTURE_DECISIONS.md")
CODEX_CONTEXT_FILE = resolve_control_path("CODEX_CONTEXT.md")

PLANNER_FILE = resolve_control_path("PLANNER.md")
EXECUTOR_FILE = resolve_control_path("EXECUTOR.md")

DEFAULT_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.4")
DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("WORKFLOW_MAX_CONTEXT_CHARS", "28000"))


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def append_markdown_section(path: pathlib.Path, heading: str, body: str) -> None:
    ensure_parent(path)
    timestamp = utc_now_iso()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {heading}\n\n")
        f.write(f"- Timestamp: {timestamp}\n\n")
        if body.strip():
            f.write(body.rstrip())
            f.write("\n")


def run_cmd(cmd: list[str], cwd: Optional[pathlib.Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        check=check,
        text=True,
        capture_output=True,
    )


def verify_git_repo() -> None:
    try:
        run_cmd(["git", "rev-parse", "--is-inside-work-tree"])
    except Exception as exc:
        raise RuntimeError("Not inside a git repository.") from exc


def git_commit(tag_label: str) -> Optional[str]:
    try:
        diff = run_cmd(["git", "status", "--porcelain"], check=False)
        if not diff.stdout.strip():
            return None
        run_cmd(["git", "add", "-A"])
        run_cmd(["git", "commit", "-m", f"workflow: {tag_label}"], check=False)
        sha = run_cmd(["git", "rev-parse", "HEAD"]).stdout.strip()
        return sha
    except Exception:
        return None


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"step": "planner"}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"step": "planner"}


def save_state(step: Step) -> None:
    ensure_parent(STATE_FILE)
    STATE_FILE.write_text(json.dumps({"step": step}, indent=2), encoding="utf-8")


def extract_status_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in STATUS_FIELD_RE.finditer(text):
        key = match.group("key").strip()
        value = match.group("value").strip()
        fields[key] = value.strip("`")
    return fields


def parse_phase_number(phase_value: str) -> int:
    phase_value = phase_value.strip()
    if phase_value.isdigit():
        return int(phase_value)

    match = re.search(r"\b(\d+)\b", phase_value)
    if match:
        return int(match.group(1))

    raise RuntimeError(f"Could not parse phase number from STATUS.md field: {phase_value!r}")


def load_status_snapshot() -> StatusSnapshot:
    text = read_text_if_exists(STATUS_FILE)
    if not text.strip():
        raise RuntimeError(f"{STATUS_FILE.name} is missing or empty.")

    fields = extract_status_fields(text)

    required = ["Phase", "Checkpoint", "Status", "Next step"]
    missing = [k for k in required if k not in fields]
    if missing:
        raise RuntimeError(f"{STATUS_FILE.name} is missing required fields: {', '.join(missing)}")

    phase_raw = fields["Phase"]
    return StatusSnapshot(
        phase_raw=phase_raw,
        phase_number=parse_phase_number(phase_raw),
        checkpoint=fields["Checkpoint"],
        status=fields["Status"],
        next_step=fields["Next step"],
    )


def replace_status_field(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^- {re.escape(key)}:.*$")
    replacement = f"- {key}: {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    if not text.endswith("\n"):
        text += "\n"
    return text + replacement + "\n"


def update_status_snapshot(
    *,
    phase: Optional[str] = None,
    checkpoint: Optional[str] = None,
    status: Optional[str] = None,
    next_step: Optional[str] = None,
    last_update: Optional[str] = None,
) -> None:
    text = read_text_if_exists(STATUS_FILE, "# STATUS.md\n\n## Workflow Status\n\n")
    updates = {
        "Phase": phase,
        "Checkpoint": checkpoint,
        "Status": status,
        "Next step": next_step,
        "Last update": last_update,
    }
    for key, value in updates.items():
        if value is None:
            continue
        text = replace_status_field(text, key, value)
    write_text_atomic(STATUS_FILE, text)


def load_phases() -> list[Phase]:
    plan_text = read_text_if_exists(PLAN_FILE)
    if not plan_text.strip():
        raise RuntimeError(f"{PLAN_FILE.name} is missing or empty.")

    phases: list[Phase] = []
    for match in PHASE_HEADING_RE.finditer(plan_text):
        phases.append(
            Phase(
                number=int(match.group("number")),
                name=match.group("title").strip(),
            )
        )

    if not phases:
        raise RuntimeError(f"No phase headings found in {PLAN_FILE.name}.")

    return phases


def find_phase(phases: list[Phase], phase_number: int) -> Phase:
    for phase in phases:
        if phase.number == phase_number:
            return phase
    raise RuntimeError(f"Phase {phase_number} not found in {PLAN_FILE.name}.")


def infer_step_from_status(snapshot: StatusSnapshot) -> Optional[Step]:
    next_step = snapshot.next_step.strip().lower()
    status = snapshot.status.strip().lower()

    if "planner" in next_step:
        return "planner"
    if "executor" in next_step or "implement" in next_step:
        return "executor"

    if status in {"ready", "pending", "planning"}:
        return "planner"
    if status in {"in_progress", "in progress", "executing", "implementation"}:
        return "executor"

    return None


def resolve_resume_step(initial_step: Optional[Step]) -> Step:
    if initial_step is not None:
        return initial_step

    saved_step = load_state().get("step", "planner")
    snapshot = load_status_snapshot()
    status_step = infer_step_from_status(snapshot)

    if status_step is None:
        return saved_step  # best effort fallback

    if status_step != saved_step:
        print(
            f"Workflow state disagrees with STATUS.md; using {status_step} "
            f"from STATUS.md instead of {saved_step}.",
            file=sys.stderr,
        )
        save_state(status_step)

    return status_step


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 64)
    return text[:keep] + "\n\n[... truncated by workflow ...]\n"


def read_context(max_context_chars: int) -> str:
    files = [
        ("AGENTS.md", AGENTS_FILE, max(1200, max_context_chars // 10)),
        ("PLAN.md", PLAN_FILE, max(3000, max_context_chars // 5)),
        ("PHASE_SPEC.md", PHASE_SPEC_FILE, max(2200, max_context_chars // 8)),
        ("ARCHITECTURE.md", ARCHITECTURE_FILE, max(1800, max_context_chars // 10)),
        ("ARCHITECTURE_DECISIONS.md", ARCHITECTURE_DECISIONS_FILE, max(1600, max_context_chars // 10)),
        ("CODEX_CONTEXT.md", CODEX_CONTEXT_FILE, max(1800, max_context_chars // 10)),
        ("CHECKLIST.md", CHECKLIST_FILE, max(2200, max_context_chars // 8)),
        ("STATUS.md", STATUS_FILE, max(1600, max_context_chars // 10)),
        ("WORKLOG.md", WORKLOG_FILE, max(2200, max_context_chars // 8)),
    ]

    sections: list[str] = []
    total = 0

    for label, path, budget in files:
        text = read_text_if_exists(path)
        if not text.strip():
            continue
        chunk = truncate(text, budget)
        block = f"===== BEGIN {label} =====\n{chunk}\n===== END {label} ====="
        if total + len(block) > max_context_chars and sections:
            break
        sections.append(block)
        total += len(block)

    return "\n\n".join(sections)


def load_prompt_file(path: pathlib.Path, role: str) -> str:
    text = read_text_if_exists(path).strip()
    if text:
        return text

    if role == "planner":
        return (
            "You are the Planner.\n"
            "Read the provided repository control documents.\n"
            "Identify the active phase from STATUS.md and PLAN.md.\n"
            "Produce a narrow, phase-bounded plan only.\n"
            "Do not implement code.\n"
            "Do not advance beyond one phase.\n"
            "If STATUS.md is READY, convert it to an executor-ready phase plan and update STATUS.md accordingly.\n"
        )

    return (
        "You are the Executor.\n"
        "Read the provided repository control documents.\n"
        "Implement only the active phase.\n"
        "Follow CHECKLIST.md.\n"
        "Update STATUS.md and WORKLOG.md.\n"
        "Perform executor self-review before finishing.\n"
        "Do not advance to the next phase automatically.\n"
    )


def build_codex_prompt(prompt_text: str, context: str, extra: str) -> str:
    return (
        f"{prompt_text.strip()}\n\n"
        f"{extra.strip()}\n\n"
        "Repository control context follows.\n\n"
        f"{context}"
    )


def detect_codex_command() -> list[str]:
    env_cmd = os.environ.get("CODEX_CMD", "").strip()
    if env_cmd:
        return env_cmd.split()

    candidates = [
        ["codex", "exec"],
        ["codex"],
    ]
    for cmd in candidates:
        try:
            probe = subprocess.run(
                cmd + ["--help"],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
            )
            if probe.returncode == 0 or probe.stdout or probe.stderr:
                return cmd
        except FileNotFoundError:
            continue

    raise RuntimeError(
        "Could not find Codex CLI. Set CODEX_CMD, for example:\n"
        "  export CODEX_CMD='codex exec'"
    )


def run_codex_exec(prompt: str, *, model: str, allow_edits: bool) -> str:
    cmd = detect_codex_command()

    base = cmd + ["--model", model]

    if allow_edits:
        base += ["--dangerously-bypass-approvals-and-sandbox"]

    proc = subprocess.run(
        base,
        input=prompt,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if proc.returncode != 0:
        raise RuntimeError(
            "Codex CLI failed.\n"
            f"Command: {' '.join(base)}\n"
            f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )

    return stdout if stdout else stderr


def run_planner(model: str, max_context_chars: int) -> str:
    prompt_text = load_prompt_file(PLANNER_FILE, "planner")
    ctx = read_context(max_context_chars)
    prompt = build_codex_prompt(
        prompt_text,
        ctx,
        extra=(
            "Task: planning only. Read STATUS.md and PLAN.md, identify the active phase, "
            "prepare a precise single-phase plan, and update STATUS.md so the next step is executor. "
            "Do not implement code."
        ),
    )
    out = run_codex_exec(prompt, model=model, allow_edits=True)
    append_markdown_section(WORKLOG_FILE, "planner-run", out or "Planner completed.")
    return out


def run_executor(model: str, max_context_chars: int) -> str:
    prompt_text = load_prompt_file(EXECUTOR_FILE, "executor")
    ctx = read_context(max_context_chars)
    prompt = build_codex_prompt(
        prompt_text,
        ctx,
        extra=(
            "Task: implementation only. Implement only the active phase, follow PHASE_SPEC.md and CHECKLIST.md, "
            "update WORKLOG.md with changed files / validations / risks, and update STATUS.md when finished. "
            "Do not move to the next phase automatically."
        ),
    )
    out = run_codex_exec(prompt, model=model, allow_edits=True)
    append_markdown_section(WORKLOG_FILE, "executor-run", out or "Executor completed.")
    return out


def workflow(*, initial_step: Optional[Step], model: str, max_context_chars: int, no_git: bool) -> None:
    if not no_git:
        verify_git_repo()

    phases = load_phases()
    step = resolve_resume_step(initial_step)

    try:
        snapshot = load_status_snapshot()
        active_phase = find_phase(phases, snapshot.phase_number)

        print(f"Resuming at: {step} (Phase {active_phase.number}: {active_phase.name} / {snapshot.checkpoint})")

        if step == "planner":
            update_status_snapshot(
                status="PLANNING",
                next_step="planner",
                last_update=today_iso(),
            )
            print("Running planner...")
            run_planner(model=model, max_context_chars=max_context_chars)
            update_status_snapshot(
                status="READY_FOR_EXECUTOR",
                next_step="executor",
                last_update=today_iso(),
            )
            save_state("executor")
            print("Next step: executor")
            return

        if step == "executor":
            update_status_snapshot(
                status="IN_PROGRESS",
                next_step="executor",
                last_update=today_iso(),
            )
            print("Running executor...")
            run_executor(model=model, max_context_chars=max_context_chars)

            # Executor must not auto-advance phase.
            # We only persist that the next action returns to planner.
            update_status_snapshot(
                status="COMPLETED_PENDING_PLANNER",
                next_step="planner",
                last_update=today_iso(),
            )

            if not no_git:
                sha = git_commit(f"phase-{active_phase.number}-executor-checkpoint")
                if sha:
                    print(f"Executor checkpoint commit: {sha}")

            save_state("planner")
            print("Next step: planner")
            return

        raise RuntimeError(f"Unsupported workflow step: {step}")

    except Exception as exc:
        print(f"Workflow failure: {exc}", file=sys.stderr)
        tb = traceback.format_exc()
        append_markdown_section(CRASH_LOG_FILE, "workflow-crash", tb)

        try:
            update_status_snapshot(
                status="BLOCKED",
                next_step="planner",
                last_update=today_iso(),
            )
        except Exception:
            pass

        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aligned planner/executor workflow for Codex.")
    parser.add_argument(
        "--step",
        choices=sorted(VALID_STEPS),
        help="Force the starting step instead of using STATUS.md / saved state.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Codex model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip git checks and git commit checkpoints.",
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
        max_context_chars=max(6000, args.max_context_chars),
        no_git=bool(args.no_git),
    )


if __name__ == "__main__":
    main()