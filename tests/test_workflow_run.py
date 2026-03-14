from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_RUN = ROOT / "workflow" / "run.py"


def load_workflow_run_module():
    module_name = "workflow_run_for_tests"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, WORKFLOW_RUN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_load_phases_accepts_live_plans_heading_style():
    workflow_run = load_workflow_run_module()

    phases = workflow_run.load_phases()

    assert [phase.number for phase in phases] == list(range(1, 9))
    assert phases[0].name == "Phase 1 - Planning/spec"
    assert phases[-1].name == "Phase 8 - Docs, examples, and test hardening"


def test_find_phase_index_falls_back_to_phase_number_when_titles_differ():
    workflow_run = load_workflow_run_module()
    phases = workflow_run.load_phases()

    phase_index = workflow_run.find_phase_index(
        phases,
        "Phase 2 — Additive Family/Template Role Metadata",
    )

    assert phase_index == 1


def test_runner_uses_repo_root_control_docs_and_prompt_files():
    workflow_run = load_workflow_run_module()

    assert workflow_run.PLANS_FILE == ROOT / "PLANS.md"
    assert workflow_run.STATUS_FILE == ROOT / "STATUS.md"
    assert workflow_run.WORKLOG_FILE == ROOT / "WORKLOG.md"
    assert workflow_run.REVIEW_FILE == ROOT / "REVIEW.md"
    assert workflow_run.PLANNER_FILE == ROOT / "PLANNER.md"
    assert workflow_run.EXECUTOR_FILE == ROOT / "EXECUTOR.md"
    assert workflow_run.REVIEWER_FILE == ROOT / "REVIEWER.md"


def test_runner_uses_state_dir_instead_of_nested_workflow_path():
    workflow_run = load_workflow_run_module()

    assert workflow_run.STATE_FILE == ROOT / "workflow" / "state" / "state.json"
    assert workflow_run.LEGACY_STATE_FILE == ROOT / "workflow" / "workflow" / "state.json"


def test_read_context_reads_repo_root_status_snapshot():
    workflow_run = load_workflow_run_module()

    context = workflow_run.read_context(12000)

    assert "===== STATUS.md =====" in context
    assert "Minimal dashboard for phased multi-role execution." in context


def test_load_status_snapshot_reads_repo_root_current_phase():
    workflow_run = load_workflow_run_module()

    snapshot = workflow_run.load_status_snapshot()

    assert snapshot.phase_name == "Phase 1 - Planning/spec"
    assert snapshot.phase_number == 1
    assert snapshot.checkpoint == "P1.0"


def test_run_codex_exec_places_never_approval_before_exec_subcommand(monkeypatch):
    workflow_run = load_workflow_run_module()
    captured = {}

    monkeypatch.setattr(workflow_run, "command_exists", lambda name: True)

    def fake_run_cmd(cmd, *, check=True, cwd=workflow_run.ROOT, env=None):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(stdout='{"approved": false}', stderr="")

    monkeypatch.setattr(workflow_run, "run_cmd", fake_run_cmd)

    workflow_run.run_codex_exec(
        "review prompt",
        model="gpt-test",
        allow_edits=False,
        use_search=False,
    )

    assert captured["cmd"][:4] == ["codex", "--ask-for-approval", "never", "exec"]
    assert "--sandbox" in captured["cmd"]
    assert "read-only" in captured["cmd"]
    assert captured["cmd"][-1] == "review prompt"


def test_run_codex_exec_keeps_edit_mode_on_exec_subcommand(monkeypatch):
    workflow_run = load_workflow_run_module()
    captured = {}

    monkeypatch.setattr(workflow_run, "command_exists", lambda name: True)

    def fake_run_cmd(cmd, *, check=True, cwd=workflow_run.ROOT, env=None):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(stdout="done", stderr="")

    monkeypatch.setattr(workflow_run, "run_cmd", fake_run_cmd)

    workflow_run.run_codex_exec(
        "executor prompt",
        model="gpt-test",
        allow_edits=True,
        use_search=False,
    )

    assert captured["cmd"][:2] == ["codex", "exec"]
    assert "--full-auto" in captured["cmd"]
    assert "--ask-for-approval" not in captured["cmd"]


def test_workflow_advances_through_successive_approved_phases(monkeypatch):
    workflow_run = load_workflow_run_module()

    current = {"phase": 1}
    calls = {"planner": 0, "executor": 0, "reviewer": 0}
    saved_steps = []
    git_tags = []
    phase_advances = []

    monkeypatch.setattr(workflow_run, "load_state", lambda: {"step": "planner"})
    monkeypatch.setattr(workflow_run, "save_state", lambda step: saved_steps.append(step))
    monkeypatch.setattr(workflow_run, "verify_git_repo", lambda: None)
    monkeypatch.setattr(
        workflow_run,
        "load_phases",
        lambda plans_path=workflow_run.PLANS_FILE: [
            workflow_run.Phase(1, "Phase 1 - Planning/spec"),
            workflow_run.Phase(2, "Phase 2 - Additive Family/Template Role Metadata"),
        ],
    )
    monkeypatch.setattr(
        workflow_run,
        "load_status_snapshot",
        lambda status_path=workflow_run.STATUS_FILE: workflow_run.StatusSnapshot(
            phase_name=(
                "Phase 1 - Planning/spec"
                if current["phase"] == 1
                else "Phase 2 - Additive Family/Template Role Metadata"
            ),
            phase_number=current["phase"],
            checkpoint=f"P{current['phase']}.0",
            status="in progress",
            next_step="planner",
        ),
    )

    def fake_run_planner(*, model, max_context_chars):
        calls["planner"] += 1
        return "planner"

    def fake_run_executor(*, model, max_context_chars):
        calls["executor"] += 1
        return "executor"

    def fake_run_reviewer(*, model, max_context_chars):
        calls["reviewer"] += 1
        return {
            "approved": True,
            "executor_can_proceed": True,
            "summary": f"phase {current['phase']} approved",
            "issues": [],
        }

    monkeypatch.setattr(workflow_run, "run_planner", fake_run_planner)
    monkeypatch.setattr(workflow_run, "run_executor", fake_run_executor)
    monkeypatch.setattr(workflow_run, "run_reviewer", fake_run_reviewer)
    monkeypatch.setattr(
        workflow_run,
        "advance_status_to_next_phase",
        lambda phases, snapshot: phase_advances.append(snapshot.phase_number)
        or current.update({"phase": snapshot.phase_number + 1})
        or workflow_run.Phase(current["phase"], "Phase 2 - Additive Family/Template Role Metadata"),
    )
    monkeypatch.setattr(
        workflow_run,
        "git_commit",
        lambda tag: git_tags.append(tag) or f"sha-{len(git_tags)}",
    )

    workflow_run.workflow(
        initial_step="planner",
        model="gpt-test",
        max_context_chars=4000,
        no_git=False,
    )

    assert calls == {"planner": 2, "executor": 2, "reviewer": 2}
    assert git_tags == [
        "phase1-executor-checkpoint",
        "phase1-checkpoint",
        "phase2-executor-checkpoint",
        "phase2-checkpoint",
    ]
    assert phase_advances == [1]
    assert saved_steps == [
        "executor",
        "reviewer",
        "planner",
        "executor",
        "reviewer",
        "planner",
    ]


def test_workflow_prefers_status_step_when_saved_state_is_stale(monkeypatch):
    workflow_run = load_workflow_run_module()

    calls = {"planner": 0, "executor": 0, "reviewer": 0}
    saved_steps = []

    monkeypatch.setattr(workflow_run, "load_state", lambda: {"step": "planner"})
    monkeypatch.setattr(workflow_run, "save_state", lambda step: saved_steps.append(step))
    monkeypatch.setattr(workflow_run, "verify_git_repo", lambda: None)
    monkeypatch.setattr(
        workflow_run,
        "load_phases",
        lambda plans_path=workflow_run.PLANS_FILE: [
            workflow_run.Phase(1, "Phase 1 - Planning/spec"),
        ],
    )
    monkeypatch.setattr(
        workflow_run,
        "load_status_snapshot",
        lambda status_path=workflow_run.STATUS_FILE: workflow_run.StatusSnapshot(
            phase_name="Phase 1 - Planning/spec",
            phase_number=1,
            checkpoint="P1.0",
            status="complete",
            next_step="reviewer validation",
        ),
    )

    monkeypatch.setattr(
        workflow_run,
        "run_planner",
        lambda **kwargs: calls.__setitem__("planner", calls["planner"] + 1),
    )
    monkeypatch.setattr(
        workflow_run,
        "run_executor",
        lambda **kwargs: calls.__setitem__("executor", calls["executor"] + 1),
    )

    def fake_run_reviewer(*, model, max_context_chars):
        calls["reviewer"] += 1
        return {
            "approved": True,
            "executor_can_proceed": True,
            "summary": "approved",
            "issues": [],
        }

    monkeypatch.setattr(workflow_run, "run_reviewer", fake_run_reviewer)
    monkeypatch.setattr(workflow_run, "git_commit", lambda tag: tag)

    workflow_run.workflow(
        initial_step=None,
        model="gpt-test",
        max_context_chars=4000,
        no_git=False,
    )

    assert calls == {"planner": 0, "executor": 0, "reviewer": 1}
    assert saved_steps == ["reviewer", "planner"]
