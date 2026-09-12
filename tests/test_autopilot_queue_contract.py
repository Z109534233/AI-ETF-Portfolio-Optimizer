import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / ".github" / "workflows" / "methodology-autopilot.yml"
TASK = ROOT / ".github" / "workflows" / "autopilot-task.yml"
ROUND = ROOT / ".github" / "workflows" / "openai-autofix-round.yml"
REVIEWER = ROOT / ".github" / "workflows" / "openai-reviewer.yml"
TASKS_DIR = ROOT / "automation" / "tasks"

PROTECTED_PREFIXES = (".github/", "automation/")


def _job_block(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"  {name}:")
    if next_name is None:
        return text[start:]
    end = text.index(f"  {next_name}:", start)
    return text[start:end]


def _load_task_spec(task_id: str) -> dict:
    return json.loads((TASKS_DIR / f"{task_id}.json").read_text(encoding="utf-8"))


# --- queue entrypoint -------------------------------------------------------


def test_queue_is_manual_workflow_dispatch_only():
    text = QUEUE.read_text(encoding="utf-8")
    on_block = text[text.index("\non:") : text.index("\njobs:")]
    assert "workflow_dispatch:" in on_block
    for forbidden_trigger in (
        "\n  push:",
        "\n  pull_request:",
        "\n  pull_request_target:",
        "\n  schedule:",
    ):
        assert forbidden_trigger not in on_block

    assert "refs/heads/main" in text
    assert "confirm" in text


def test_queue_requires_main_ref_guard():
    text = QUEUE.read_text(encoding="utf-8")
    guard = _job_block(text, "guard", "create_integration_branch")
    assert "refs/heads/main" in guard
    assert "exit 1" in guard


# --- M1 -> M2 -> M3 -> M4 dependency chain ----------------------------------


def test_queue_jobs_form_sequential_dependency_chain():
    text = QUEUE.read_text(encoding="utf-8")
    m1 = _job_block(text, "m1", "m2")
    m2 = _job_block(text, "m2", "m3")
    m3 = _job_block(text, "m3", "m4")
    m4 = _job_block(text, "m4", "create_cumulative_pr")

    assert "needs: [guard, create_integration_branch]" in m1
    assert "needs: [guard, m1]" in m2 and "needs.m1.outputs.status == 'merged'" in m2
    assert "needs: [guard, m2]" in m3 and "needs.m2.outputs.status == 'merged'" in m3
    assert "needs: [guard, m3]" in m4 and "needs.m3.outputs.status == 'merged'" in m4

    for block, task_id in ((m1, "M1"), (m2, "M2"), (m3, "M3"), (m4, "M4")):
        assert f"task_id: {task_id}" in block
        assert "uses: ./.github/workflows/autopilot-task.yml" in block


def test_task_spec_dependency_chain_matches():
    m1 = _load_task_spec("M1")
    m2 = _load_task_spec("M2")
    m3 = _load_task_spec("M3")
    m4 = _load_task_spec("M4")

    assert m1["depends_on"] is None
    assert m2["depends_on"] == "M1"
    assert m3["depends_on"] == "M2"
    assert m4["depends_on"] == "M3"


# --- task spec safety --------------------------------------------------------


def test_all_four_task_specs_exist_with_required_fields():
    for task_id in ("M1", "M2", "M3", "M4"):
        spec = _load_task_spec(task_id)
        assert spec["task_id"] == task_id
        assert spec["title"]
        assert spec["context_files"]
        assert spec["allowed_files"]
        assert spec["acceptance_criteria"]


def test_task_specs_cannot_allow_github_or_automation_paths():
    for task_id in ("M1", "M2", "M3", "M4"):
        spec = _load_task_spec(task_id)
        for key in ("context_files", "allowed_files"):
            for path in spec[key]:
                assert not path.startswith(PROTECTED_PREFIXES), (
                    f"{task_id}.{key} allows protected path: {path}"
                )
                assert ".." not in Path(path).parts
                assert not Path(path).is_absolute()


# --- initial Claude implementation job is read-only/tool-less/pinned -------


def test_claude_job_is_toolless_read_only_and_pinned():
    text = TASK.read_text(encoding="utf-8")
    block = _job_block(text, "claude_patch", "validate_patch")
    assert "contents: read" in block
    assert "contents: write" not in block
    assert (
        "uses: anthropics/claude-code-action@56cf60fde42f7b19c3abfd5c9c48b69a1288461f"
        in block
    )
    assert "github_token: ${{ github.token }}" in block
    assert "--permission-mode dontAsk" in block
    assert '--tools ""' in block
    assert "--output-format json" in block
    assert '"patch"' in block and '"summary"' in block
    assert "CLAUDE_CODE_OAUTH_TOKEN" in block


# --- write/apply job has no AI secrets --------------------------------------


def test_apply_patch_job_has_no_ai_secrets():
    text = TASK.read_text(encoding="utf-8")
    block = _job_block(text, "apply_patch", "test")
    assert "contents: write" in block
    assert "OPENAI_API_KEY" not in block
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in block


def test_test_job_has_no_ai_secrets():
    text = TASK.read_text(encoding="utf-8")
    block = _job_block(text, "test", "create_pr")
    assert "OPENAI_API_KEY" not in block
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in block
    assert "tests/pages" in block


def test_validate_patch_job_has_no_ai_secrets_and_enforces_allowlist():
    text = TASK.read_text(encoding="utf-8")
    block = _job_block(text, "validate_patch", "apply_patch")
    assert "OPENAI_API_KEY" not in block
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in block
    assert "'.github/workflows/'" in block
    assert "'.github/actions/'" in block
    assert "'automation/'" in block
    assert "escaped the task-spec allowlist" in block
    assert "git', 'apply', '--check'" in block


# --- up to 3 OpenAI repair rounds per task ----------------------------------


def test_task_pipeline_has_exactly_three_repair_rounds():
    text = TASK.read_text(encoding="utf-8")
    for round_name in ("round_1", "round_2", "round_3"):
        assert f"\n  {round_name}:" in text
    assert "\n  round_4:" not in text
    assert text.count("uses: ./.github/workflows/openai-autofix-round.yml") == 3

    round_2 = _job_block(text, "round_2", "round_3")
    round_3 = _job_block(text, "round_3", "finalize_verdict")
    assert "< 3" in round_2
    assert "< 3" in round_3


def test_cumulative_pr_also_capped_at_three_repair_rounds():
    text = QUEUE.read_text(encoding="utf-8")
    for round_name in ("cumulative_round_1", "cumulative_round_2", "cumulative_round_3"):
        assert f"\n  {round_name}:" in text
    assert "\n  cumulative_round_4:" not in text


# --- task auto-merge guard ---------------------------------------------------


def test_task_merge_gate_has_explicit_integration_branch_guard():
    text = TASK.read_text(encoding="utf-8")
    block = _job_block(text, "merge_gate", "report_status")
    assert "if: needs.finalize_verdict.outputs.final_verdict == 'PASS'" in block
    assert 'if [ "$INTEGRATION_BRANCH" = "main" ]' in block
    assert "integrationBranch === 'main'" in block
    assert "pulls.merge" in block


def test_guard_integration_refuses_main_as_integration_branch():
    text = TASK.read_text(encoding="utf-8")
    block = _job_block(text, "guard_integration", "create_task_branch")
    assert "inputs.integration_branch }}\" = \"main\"" in block
    assert "exit 1" in block


# --- final cumulative PR has no merge-to-main action ------------------------


def test_cumulative_pr_workflow_never_merges_to_main():
    text = QUEUE.read_text(encoding="utf-8")
    assert "pulls.merge" not in text
    assert "merge_method" not in text
    assert "base: 'main'" in text
    assert "head: integrationBranch" in text


# --- normal reviewer + trusted_automation policy ----------------------------


def test_normal_reviewer_still_enabled_for_non_autopilot_prs():
    text = REVIEWER.read_text(encoding="utf-8")
    assert "pull_request_target:" in text
    assert "types: [opened, synchronize, reopened, ready_for_review]" in text
    # the autopilot skip must be scoped to autopilot/ head refs, not a blanket disable
    assert "pr.head.ref.startsWith('autopilot/')" in text
    assert "OWNER" in text and "MEMBER" in text and "COLLABORATOR" in text


def test_trusted_automation_defaults_false_and_does_not_weaken_normal_policy():
    text = ROUND.read_text(encoding="utf-8")
    inputs_block = text[text.index("workflow_call:") : text.index("secrets:")]
    assert "trusted_automation:" in inputs_block
    assert "default: false" in inputs_block

    reserve = _job_block(text, "reserve", "prepare_context")
    # normal (non-trusted) path must still require OWNER/MEMBER/COLLABORATOR
    assert "['OWNER', 'MEMBER', 'COLLABORATOR'].includes(pr.author_association" in reserve
    # trusted path is additive and requires an autopilot/ head ref plus bot authorship
    assert "trustedAutomation" in reserve
    assert "github-actions[bot]" in reserve
    assert "pr.head.ref.startsWith('autopilot/')" in reserve
    # a normal call can never satisfy an autopilot-managed controller state or vice versa
    assert "trustedAutomation !== (state.autopilot_managed === true)" in reserve
    # the same-repository check is unconditional for both paths
    same_repo_idx = reserve.index("Auto-fix is restricted to same-repository PRs")
    trusted_branch_idx = reserve.index("if (trustedAutomation) {")
    assert same_repo_idx < trusted_branch_idx
