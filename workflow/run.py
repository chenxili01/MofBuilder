#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Sequence

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_phases as shared


PLAN_MODEL = "gpt-5.4"
EXECUTOR_MODEL = "gpt-5.4"
REVIEW_MODEL = "gpt-5.4"
PLAN_EFFORT = "high"
EXECUTOR_EFFORT = "medium"
REVIEW_EFFORT = "high"

RUNNER_STATE_FILE = shared.STATE_DIR / "run_state.json"
SDK_RUNNER = shared.WORKFLOW_DIR / "codex_sdk_runner.mjs"

PHASE_PATTERN = re.compile(r"^## (Phase (\d+) [^\n]+)$", re.MULTILINE)
STATUS_PHASE_PATTERN = re.compile(r"^- Phase:\s+`([^`]+)`\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Phase:
    index: int
    number: int
    name: str

    @property
    def checkpoint(self) -> str:
        return f"P{self.number}.0"


@dataclass(frozen=True)
class RoleConfig:
    role: str
    prompt_file: pathlib.Path
    model: str
    reasoning_effort: str


@dataclass(frozen=True)
class RunnerConfig:
    no_git: bool
    max_rounds: int
    phase_limit: int | None


ROLE_CONFIGS = {
    "planner": RoleConfig("planner", shared.PROMPT_FILES["planner"], PLAN_MODEL, PLAN_EFFORT),
    "executor": RoleConfig(
        "executor", shared.PROMPT_FILES["executor"], EXECUTOR_MODEL, EXECUTOR_EFFORT
    ),
    "reviewer": RoleConfig(
        "reviewer", shared.PROMPT_FILES["reviewer"], REVIEW_MODEL, REVIEW_EFFORT
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Run the full PLANS.md phase workflow.")
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip automatic git checkpoint commits.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="Maximum planner/executor/reviewer rounds allowed per phase.",
    )
    parser.add_argument(
        "--phase-limit",
        type=int,
        default=None,
        help="Optional number of remaining phases to run before stopping.",
    )
    args = parser.parse_args(argv)

    if args.max_rounds <= 0:
        raise shared.WorkflowError("--max-rounds must be a positive integer.")
    if args.phase_limit is not None and args.phase_limit <= 0:
        raise shared.WorkflowError("--phase-limit must be a positive integer when provided.")

    return RunnerConfig(
        no_git=args.no_git,
        max_rounds=args.max_rounds,
        phase_limit=args.phase_limit,
    )


def load_phases() -> list[Phase]:
    plans_text = shared.read_required_text(shared.PLANS, "PLANS.md")
    phases = []
    for index, match in enumerate(PHASE_PATTERN.finditer(plans_text)):
        phases.append(Phase(index=index, number=int(match.group(2)), name=match.group(1)))
    if not phases:
        raise shared.WorkflowError("No phase headings were found in PLANS.md.")
    return phases


def current_status_phase() -> str | None:
    status_text = shared.read_text(shared.STATUS, default="")
    match = STATUS_PHASE_PATTERN.search(status_text)
    if not match:
        return None
    return match.group(1).strip()


def find_phase_index(phases: list[Phase], phase_name: str | None) -> int:
    if phase_name:
        for phase in phases:
            if phase.name == phase_name:
                return phase.index
    return 0


def load_runner_state(phases: list[Phase]) -> dict[str, int]:
    path = RUNNER_STATE_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            shared.backup_corrupt_state(path, "invalid JSON")
        else:
            if isinstance(data, dict):
                phase_index = data.get("phase_index")
                round_number = data.get("round_number")
                if (
                    isinstance(phase_index, int)
                    and isinstance(round_number, int)
                    and 0 <= phase_index <= len(phases)
                    and round_number >= 1
                ):
                    return {"phase_index": phase_index, "round_number": round_number}
            shared.backup_corrupt_state(path, "invalid runner state shape")

    phase_index = find_phase_index(phases, current_status_phase())
    return {"phase_index": phase_index, "round_number": 1}


def save_runner_state(phase_index: int, round_number: int) -> None:
    payload = {"phase_index": phase_index, "round_number": round_number}
    shared.atomic_write_text(RUNNER_STATE_FILE, json.dumps(payload, indent=2) + "\n")


def update_status_for_phase(phase: Phase) -> None:
    status_text = shared.read_required_text(shared.STATUS, "STATUS.md")
    replacements = {
        "Phase": f"`{phase.name}`",
        "Checkpoint": f"`{phase.checkpoint}`",
        "Status": "pending",
        "Next step": "planner",
        "Execution mode": "automated phase runner",
    }

    updated = status_text
    for key, value in replacements.items():
        pattern = re.compile(rf"^- {re.escape(key)}:.*$", re.MULTILINE)
        replacement = f"- {key}: {value}"
        if pattern.search(updated):
            updated = pattern.sub(replacement, updated, count=1)
        else:
            updated = updated.rstrip() + f"\n{replacement}\n"

    if updated != status_text:
        shared.atomic_write_text(shared.STATUS, updated)


def build_role_prompt(phase: Phase, round_number: int, role_config: RoleConfig) -> str:
    base_prompt = shared.read_required_text(role_config.prompt_file, f"{role_config.role} prompt").strip()
    extra_lines = [
        f"Workflow runner phase: {phase.name}",
        f"Workflow runner round: {round_number}",
        f"Repository root: {shared.ROOT}",
    ]

    if role_config.role == "planner":
        extra_lines.append(
            "Operate on the current phase only. If the latest review failed, produce a remediation plan for the same phase before any forward progress."
        )
    elif role_config.role == "executor":
        extra_lines.append(
            "Execute only the current phase. Do not advance to the next phase even if the implementation appears complete."
        )
    else:
        extra_lines.extend(
            [
                "Return only one JSON object and do not edit REVIEW.md; the workflow runner will save your review.",
                shared.REVIEW_JSON_INSTRUCTIONS,
            ]
        )

    return f"{base_prompt}\n\nWorkflow runner instructions\n" + "\n".join(extra_lines)


def write_temp_prompt(prompt: str) -> pathlib.Path:
    temp_dir = shared.STATE_DIR / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=temp_dir,
        prefix="prompt-",
        suffix=".md",
        delete=False,
    ) as handle:
        handle.write(prompt)
        return pathlib.Path(handle.name)


def run_role(phase: Phase, round_number: int, role: str) -> str:
    role_config = ROLE_CONFIGS[role]
    prompt_path = write_temp_prompt(build_role_prompt(phase, round_number, role_config))
    try:
        result = subprocess.run(
            [
                "node",
                str(SDK_RUNNER),
                "--cwd",
                str(shared.ROOT),
                "--prompt-file",
                str(prompt_path),
                "--model",
                role_config.model,
                "--reasoning-effort",
                role_config.reasoning_effort,
            ],
            cwd=shared.ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            timeout=shared.REQUEST_TIMEOUT_SECONDS,
            check=False,
        )
    finally:
        prompt_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise shared.WorkflowError(
            f"{role} SDK run failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout or '[empty]'}\n"
            f"stderr:\n{result.stderr or '[empty]'}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise shared.WorkflowError(
            f"{role} SDK bridge returned invalid JSON.\nstdout:\n{result.stdout}"
        ) from exc

    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise shared.WorkflowError(f"{role} SDK bridge returned empty text output.")
    return text.strip()


def run_reviewer(phase: Phase, round_number: int) -> shared.ReviewResult:
    try:
        return shared.parse_review_result(run_role(phase, round_number, "reviewer"))
    except shared.ReviewerOutputError as exc:
        corrected_prompt = build_role_prompt(phase, round_number, ROLE_CONFIGS["reviewer"])
        corrected_prompt += (
            "\n\nThe previous reviewer output was invalid JSON. "
            f"Return only valid JSON that matches the required schema. Error: {exc}"
        )
        prompt_path = write_temp_prompt(corrected_prompt)
        try:
            result = subprocess.run(
                [
                    "node",
                    str(SDK_RUNNER),
                    "--cwd",
                    str(shared.ROOT),
                    "--prompt-file",
                    str(prompt_path),
                    "--model",
                    REVIEW_MODEL,
                    "--reasoning-effort",
                    REVIEW_EFFORT,
                ],
                cwd=shared.ROOT,
                text=True,
                capture_output=True,
                encoding="utf-8",
                timeout=shared.REQUEST_TIMEOUT_SECONDS,
                check=False,
            )
        finally:
            prompt_path.unlink(missing_ok=True)

        if result.returncode != 0:
            raise shared.WorkflowError(
                f"reviewer SDK retry failed with exit code {result.returncode}\n"
                f"stdout:\n{result.stdout or '[empty]'}\n"
                f"stderr:\n{result.stderr or '[empty]'}"
            ) from exc

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as retry_exc:
            raise shared.WorkflowError(
                f"reviewer SDK retry returned invalid JSON.\nstdout:\n{result.stdout}"
            ) from retry_exc

        retry_text = payload.get("text", "")
        if not isinstance(retry_text, str) or not retry_text.strip():
            raise shared.WorkflowError("reviewer SDK retry returned empty text output.") from exc
        return shared.parse_review_result(retry_text.strip())


def record_reviewer_output(review: shared.ReviewResult) -> None:
    shared.append_text_atomic(shared.REVIEW, shared.format_review_entry(review))


def git_checkpoint(config: RunnerConfig, tag: str) -> None:
    if config.no_git:
        return
    shared.git_commit(tag)


def run_phase(config: RunnerConfig, phase: Phase, starting_round: int) -> bool:
    update_status_for_phase(phase)
    round_number = starting_round

    while round_number <= config.max_rounds:
        save_runner_state(phase.index, round_number)
        print(f"Phase {phase.number} round {round_number}: planner")
        run_role(phase, round_number, "planner")

        print(f"Phase {phase.number} round {round_number}: executor")
        run_role(phase, round_number, "executor")
        git_checkpoint(config, f"phase-{phase.number:02d}-round-{round_number:02d}-executor")

        print(f"Phase {phase.number} round {round_number}: reviewer")
        review = run_reviewer(phase, round_number)
        record_reviewer_output(review)

        if review.approved:
            git_checkpoint(config, f"phase-{phase.number:02d}-approved")
            return True

        round_number += 1

    raise shared.WorkflowError(
        f"Phase {phase.name} exceeded the maximum round count ({config.max_rounds})."
    )


def workflow(config: RunnerConfig) -> None:
    phases = load_phases()
    state = load_runner_state(phases)
    phase_index = state["phase_index"]
    round_number = state["round_number"]
    completed = 0

    if phase_index >= len(phases):
        print("All phases already completed.")
        return

    while phase_index < len(phases):
        phase = phases[phase_index]
        print(f"Running {phase.name} starting at round {round_number}")
        approved = run_phase(config, phase, round_number)
        if not approved:
            raise shared.WorkflowError(f"Phase {phase.name} did not reach approval.")

        phase_index += 1
        completed += 1
        round_number = 1
        save_runner_state(phase_index, round_number)

        if config.phase_limit is not None and completed >= config.phase_limit:
            return


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        workflow(config)
    except shared.WorkflowError as exc:
        shared.stderr(f"Workflow failure: {exc}")
        return 1
    except KeyboardInterrupt:
        shared.stderr("Workflow interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
