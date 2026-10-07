from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / ".github" / "workflows" / "openai-reviewer.yml"
ROUND = ROOT / ".github" / "workflows" / "openai-autofix-round.yml"


def _job_block(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"  {name}:")
    if next_name is None:
        return text[start:]
    end = text.index(f"  {next_name}:", start)
    return text[start:end]


def test_entry_is_manual_dispatch_only_when_reviewer_is_disabled():
    text = ENTRY.read_text(encoding="utf-8")
    on_block = text[text.index("\non:") : text.index("\nconcurrency:")]
    assert "workflow_dispatch:" in on_block
    assert "pull_request_target:" not in on_block
    assert "\n  pull_request:\n" not in on_block
    assert "Automatic OpenAI PR review is disabled." in text
    assert "cancel-in-progress: false" in text


def test_claude_job_is_toolless_and_read_only():
    text = ROUND.read_text(encoding="utf-8")
    block = _job_block(text, "claude_patch", "validate_patch")
    assert "contents: read" in block
    assert "contents: write" not in block
    assert "github_token: ${{ github.token }}" in block
    assert "--bare" not in block  # OAuth subscription auth must remain available
    assert "--permission-mode dontAsk" in block
    assert '--tools ""' in block
    assert "CLAUDE_CODE_OAUTH_TOKEN" in block


def test_write_job_has_no_ai_credentials():
    text = ROUND.read_text(encoding="utf-8")
    block = _job_block(text, "apply_patch", "test_patch")
    assert "contents: write" in block
    assert "OPENAI_API_KEY" not in block
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in block


def test_test_job_has_no_ai_credentials():
    text = ROUND.read_text(encoding="utf-8")
    block = _job_block(text, "test_patch", "post_review")
    assert "contents: read" in block
    assert "OPENAI_API_KEY" not in block
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in block


def test_patch_validator_blocks_protected_paths_and_scope_escape():
    text = ROUND.read_text(encoding="utf-8")
    block = _job_block(text, "validate_patch", "apply_patch")
    assert "'.github/workflows/'" in block
    assert "'.github/actions/'" in block
    assert "'automation/'" in block
    assert "Patch escaped the reviewer-derived allowlist" in block
    assert "git', 'apply', '--check'" in block
