"""OpenAI-powered PR reviewer for the AI ETF Portfolio Optimizer.

Reads the PR title/body, the changed-file list, and the git diff between
base and head, then asks an OpenAI model to review the change against
automation/REVIEW_RULES.md. Produces:

- a machine-readable JSON verdict (PASS / FIX / STOP with structured issues)
- a human-readable Markdown PR comment

This script does not post to GitHub itself; the calling workflow is
responsible for taking output-comment.md and posting/updating a PR comment.

Required environment variable:
    OPENAI_API_KEY  (never hard-code this)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
MAX_DIFF_CHARS = 120_000

RESULT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["PASS", "FIX", "STOP"],
        },
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                    },
                    "file": {"type": "string"},
                    "problem": {"type": "string"},
                    "required_fix": {"type": "string"},
                },
                "required": ["severity", "file", "problem", "required_fix"],
            },
        },
        "next_instruction": {"type": "string"},
    },
    "required": ["verdict", "summary", "issues", "next_instruction"],
}


def read_text(path: str | None, default: str = "") -> str:
    if not path:
        return default
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8", errors="replace")


def truncate_diff(diff_text: str, max_chars: int) -> tuple[str, bool]:
    if len(diff_text) <= max_chars:
        return diff_text, False
    return diff_text[:max_chars], True


def build_prompt(
    pr_title: str,
    pr_body: str,
    changed_files: str,
    diff_text: str,
    diff_truncated: bool,
    rules_text: str,
) -> str:
    truncation_notice = (
        "\n\n[NOTE: The diff below was TRUNCATED to fit a size limit. "
        "It does not contain the full change. Do not assume anything about "
        "code outside the shown portion; flag reduced confidence if this "
        "materially affects your ability to judge correctness.]\n\n"
        if diff_truncated
        else "\n\n"
    )

    return f"""You are reviewing a pull request for an AI ETF Portfolio Optimizer
(a Streamlit app that performs portfolio optimization over ETF data).

Follow the review rules below exactly. They define your priorities and
what verdict to assign.

=== REVIEW RULES ===
{rules_text}
=== END REVIEW RULES ===

=== PR TITLE ===
{pr_title}

=== PR BODY ===
{pr_body if pr_body.strip() else "(no PR body provided)"}

=== CHANGED FILES ===
{changed_files if changed_files.strip() else "(no changed-file list provided)"}
{truncation_notice}=== GIT DIFF (base...head) ===
{diff_text if diff_text.strip() else "(no diff provided)"}
=== END GIT DIFF ===

Review this change and return your findings using the structured JSON
schema you have been given. Be specific: every issue must name a file
(or "unknown" if genuinely not attributable) and describe a concrete
problem and a concrete required fix. Do not pad the issues list with
stylistic nitpicks that the rules do not ask for. Do not include your
internal reasoning or chain-of-thought in any field — only conclusions.
"""


def call_openai(client: OpenAI, prompt: str) -> dict:
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "pr_review_result",
                "schema": RESULT_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = getattr(response, "output_text", None)
    if not raw:
        raise RuntimeError("OpenAI response did not contain output_text")

    return json.loads(raw)


def render_comment(result: dict) -> str:
    verdict = result.get("verdict", "STOP")
    summary = result.get("summary", "").strip() or "(no summary provided)"
    issues = result.get("issues", [])
    next_instruction = result.get("next_instruction", "").strip()

    lines = ["## OpenAI Reviewer", "", f"Verdict: {verdict}", "", "### Summary", summary, ""]

    lines.append("### Findings")
    if not issues:
        lines.append("No issues reported.")
    else:
        for issue in issues:
            severity = issue.get("severity", "unknown")
            file_ = issue.get("file", "unknown")
            problem = issue.get("problem", "")
            required_fix = issue.get("required_fix", "")
            lines.append(f"- **[{severity}] {file_}** — {problem}")
            lines.append(f"  - Required fix: {required_fix}")
    lines.append("")

    lines.append("### Required next action")
    if verdict == "PASS":
        lines.append("None — implementation satisfies the task.")
    elif next_instruction:
        lines.append(next_instruction)
    else:
        lines.append("(none provided)")
    lines.append("")

    lines.append(f"_Model: {MODEL}_")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-title", default="", help="Pull request title")
    parser.add_argument(
        "--pr-body-file", default=None, help="Path to a file containing the PR body"
    )
    parser.add_argument(
        "--changed-files-file",
        default=None,
        help="Path to a newline-delimited list of changed files",
    )
    parser.add_argument(
        "--diff-file", default=None, help="Path to a file containing the git diff"
    )
    parser.add_argument(
        "--rules-file",
        default=str(Path(__file__).parent / "REVIEW_RULES.md"),
        help="Path to REVIEW_RULES.md",
    )
    parser.add_argument(
        "--output-json",
        default="review_result.json",
        help="Where to write the machine-readable verdict JSON",
    )
    parser.add_argument(
        "--output-comment",
        default="review_comment.md",
        help="Where to write the human-readable PR comment Markdown",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

    pr_body = read_text(args.pr_body_file)
    changed_files = read_text(args.changed_files_file)
    raw_diff = read_text(args.diff_file)
    rules_text = read_text(args.rules_file)

    if not rules_text.strip():
        print(f"ERROR: rules file not found or empty: {args.rules_file}", file=sys.stderr)
        return 1

    diff_text, diff_truncated = truncate_diff(raw_diff, MAX_DIFF_CHARS)

    prompt = build_prompt(
        pr_title=args.pr_title,
        pr_body=pr_body,
        changed_files=changed_files,
        diff_text=diff_text,
        diff_truncated=diff_truncated,
        rules_text=rules_text,
    )

    client = OpenAI(api_key=api_key)

    try:
        result = call_openai(client, prompt)
    except Exception as exc:  # noqa: BLE001 - surface any API/SDK failure clearly
        print(f"ERROR: OpenAI review call failed: {exc}", file=sys.stderr)
        return 1

    Path(args.output_json).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(args.output_comment).write_text(render_comment(result), encoding="utf-8")

    print(f"Verdict: {result.get('verdict', 'UNKNOWN')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
