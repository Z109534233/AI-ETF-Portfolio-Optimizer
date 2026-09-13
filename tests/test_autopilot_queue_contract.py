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


# --- execution-critical needs/output wiring ----------------------------------

import re


def _declared_needs(block: str) -> set[str]:
    match = re.search(r"(?m)^    needs:\s*(.+)$", block)
    if not match:
        return set()
    raw = match.group(1).strip()
    if raw.startswith("[") and raw.endswith("]"):
        return {item.strip() for item in raw[1:-1].split(",") if item.strip()}
    return {raw}


def _referenced_needs(block: str) -> set[str]:
    return set(re.findall(r"needs\.([A-Za-z0-9_-]+)\.", block))


def test_round_and_finalize_jobs_only_reference_direct_needs():
    text = TASK.read_text(encoding="utf-8")
    blocks = {
        "round_1": _job_block(text, "round_1", "round_2"),
        "round_2": _job_block(text, "round_2", "round_3"),
        "round_3": _job_block(text, "round_3", "finalize_verdict"),
        "finalize_verdict": _job_block(text, "finalize_verdict", "merge_gate"),
    }
    for name, block in blocks.items():
        missing = _referenced_needs(block) - _declared_needs(block)
        assert not missing, f"{name} references non-direct needs jobs: {sorted(missing)}"


def test_create_pr_exports_every_output_consumed_by_review_rounds():
    text = TASK.read_text(encoding="utf-8")
    create_pr = _job_block(text, "create_pr", "round_1")
    outputs_section = create_pr[
        create_pr.index("    outputs:") : create_pr.index("    steps:")
    ]
    for required in (
        "pr_number:",
        "base_sha:",
        "pr_title:",
        "pr_body:",
        "controller_comment_id:",
    ):
        assert required in outputs_section

    for round_name, next_name in (
        ("round_1", "round_2"),
        ("round_2", "round_3"),
        ("round_3", "finalize_verdict"),
    ):
        block = _job_block(text, round_name, next_name)
        for output_name in (
            "pr_number",
            "base_sha",
            "pr_title",
            "pr_body",
            "controller_comment_id",
        ):
            assert f"needs.create_pr.outputs.{output_name}" in block


def test_autopilot_prefix_alone_cannot_skip_standard_reviewer():
    text = REVIEWER.read_text(encoding="utf-8")
    assert "const trustedAutopilotPr =" in text
    assert "methodology-autopilot-managed:v1" in text
    assert "pr.user.login === 'github-actions[bot]'" in text
    assert "pr.body || ''" in text
    assert "branchMatchesRun" in text
    assert "if (trustedAutopilotPr)" in text


def test_queue_marker_is_present_atomically_in_pr_creation_body():
    task_text = TASK.read_text(encoding="utf-8")
    queue_text = QUEUE.read_text(encoding="utf-8")
    assert "methodology-autopilot-managed:v1" in _job_block(task_text, "create_pr", "round_1")
    assert "methodology-autopilot-managed:v1" in _job_block(
        queue_text, "create_cumulative_pr", "cumulative_round_1"
    )


def test_every_trusted_task_repair_round_receives_task_allowlist():
    text = TASK.read_text(encoding="utf-8")
    create_pr = _job_block(text, "create_pr", "round_1")
    assert "allowed_files_json:" in create_pr
    assert "JSON.stringify(manifest.allowed_files || [])" in create_pr
    for round_name, next_name in (
        ("round_1", "round_2"),
        ("round_2", "round_3"),
        ("round_3", "finalize_verdict"),
    ):
        block = _job_block(text, round_name, next_name)
        assert "trusted_automation: true" in block
        assert (
            "task_allowed_files_json: ${{ needs.create_pr.outputs.allowed_files_json }}"
            in block
        )


def test_reusable_repair_enforces_trusted_task_allowlist():
    text = ROUND.read_text(encoding="utf-8")
    assert "task_allowed_files_json:" in text
    reserve = _job_block(text, "reserve", "prepare_context")
    validate = _job_block(text, "validate_patch", "apply_patch")
    assert "trusted_automation repair requires a non-empty task allowlist" in reserve
    assert "TASK_ALLOWED_FILES_JSON" in validate
    assert "Patch escaped the trusted task allowlist" in validate


def test_cumulative_repair_rounds_receive_union_allowlist():
    text = QUEUE.read_text(encoding="utf-8")
    create = _job_block(text, "create_cumulative_pr", "cumulative_round_1")
    assert "cumulativeAllowed" in create
    assert "allowed_files_json:" in create
    for round_name, next_name in (
        ("cumulative_round_1", "cumulative_round_2"),
        ("cumulative_round_2", "cumulative_round_3"),
        ("cumulative_round_3", None),
    ):
        block = _job_block(text, round_name, next_name)
        assert "task_allowed_files_json:" in block


def test_required_task_context_fails_closed_instead_of_silent_skip():
    text = TASK.read_text(encoding="utf-8")
    block = _job_block(text, "prepare_context", "claude_patch")
    assert "Required task context file is missing" in block
    assert "Required task context file is binary" in block
    assert "Required task context file exceeds 180 KB limit" in block
    required_loop = block[block.index("for path in context_files:") :]
    assert "if not fp.is_file():\n                  continue" not in required_loop
    assert "if b'\\0' in data:\n                  continue" not in required_loop


def test_autopilot_does_not_claim_unused_bubblewrap_sandbox():
    task_text = TASK.read_text(encoding="utf-8")
    doc_text = (ROOT / "automation" / "AUTOPILOT_QUEUE.md").read_text(encoding="utf-8")
    assert "bubblewrap" not in task_text
    assert "not an OS/container" in doc_text
    assert "read-only repository permission" in doc_text


def test_runtime_protected_prefix_policy_matches_contract():
    text = TASK.read_text(encoding="utf-8")
    prepare = _job_block(text, "prepare_context", "claude_patch")
    validate = _job_block(text, "validate_patch", "apply_patch")
    expected = "protected = ('.github/', 'automation/')"
    assert expected in prepare
    assert expected in validate
    assert ".github/workflows/" not in prepare.split("protected =", 1)[1].split("\n", 1)[0]
    assert ".github/actions/" not in validate.split("protected =", 1)[1].split("\n", 1)[0]


def test_merge_gate_requires_confirmed_merge_and_verified_branch_head():
    text = TASK.read_text(encoding="utf-8")
    block = _job_block(text, "merge_gate", "report_status")
    assert "const result = await github.rest.pulls.merge" in block
    assert "if (!result.data.merged)" in block
    assert "mergedBranch.data.commit.sha !== result.data.sha" in block
    assert "Task PR merge confirmed at integration SHA" in block


def test_trusted_repair_validator_executes_with_valid_allowlist(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    workflow = ROUND.read_text(encoding="utf-8")
    block = _job_block(workflow, "validate_patch", "apply_patch")
    marker = "python - <<'PY'\n"
    start = block.index(marker) + len(marker)
    end = block.index("\n          PY", start)
    script = textwrap.dedent(block[start:end])

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Autofix Test"],
        cwd=tmp_path,
        check=True,
    )

    target = tmp_path / "example.txt"
    target.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "example.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=tmp_path,
        check=True,
    )

    (tmp_path / "repair_context").mkdir()
    (tmp_path / "patch_artifact").mkdir()
    (tmp_path / "repair_context" / "manifest.json").write_text(
        json.dumps({"allowed_files": ["example.txt"]}),
        encoding="utf-8",
    )
    patch = (
        "--- a/example.txt\n"
        "+++ b/example.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    (tmp_path / "patch_artifact" / "claude_patch.json").write_text(
        json.dumps({"patch": patch, "summary": "fixture"}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["TRUSTED_AUTOMATION"] = "true"
    env["TASK_ALLOWED_FILES_JSON"] = json.dumps(["example.txt"])

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "validated.patch").is_file()
    assert target.read_text(encoding="utf-8") == "new\n"


def test_every_repair_round_tests_repaired_head_before_post_review():
    text = ROUND.read_text(encoding="utf-8")
    test_patch = _job_block(text, "test_patch", "post_review")
    post_review = _job_block(text, "post_review", "finalize")
    finalize = _job_block(text, "finalize")

    assert "ref: ${{ needs.apply_patch.outputs.head_sha }}" in test_patch
    assert "python -m pytest -q" in test_patch
    assert "if: needs.test_patch.result == 'success'" in post_review
    assert "process.env.TEST_RESULT === 'success'" in finalize
    assert "tested_head_sha" in finalize


def test_task_never_advances_or_merges_unverified_repair_head():
    text = TASK.read_text(encoding="utf-8")
    round_2 = _job_block(text, "round_2", "round_3")
    round_3 = _job_block(text, "round_3", "finalize_verdict")
    merge = _job_block(text, "merge_gate", "report_status")

    assert (
        "needs.round_1.outputs.tested_head_sha == needs.round_1.outputs.head_sha"
        in round_2
    )
    assert (
        "needs.round_2.outputs.tested_head_sha == needs.round_2.outputs.head_sha"
        in round_3
    )
    assert "EXPECTED_TESTED_HEAD_SHA" in merge
    assert "EXPECTED_TESTED_HEAD_SHA !== process.env.EXPECTED_HEAD_SHA" in merge
