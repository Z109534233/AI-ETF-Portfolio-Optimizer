# OpenAI → Claude Auto-fix Security Boundary

This repository uses a deliberately split workflow for automatic PR repair.

## Trust model

Automatic repair is allowed only when all of these are true:

- the PR head is in this repository (forks are review-only);
- the PR author association is `OWNER`, `MEMBER`, or `COLLABORATOR`;
- the PR does not modify `.github/workflows/`, `.github/actions/`, or `automation/`;
- the persistent controller is not `blocked`, `exhausted`, or `reserved`;
- fewer than three automatic repair attempts have been consumed.

The entry workflow uses `pull_request_target`, so the workflow definition comes
from the trusted base branch. PR code is never executed by jobs holding OpenAI
or Claude credentials.

## Credential separation

### OpenAI review jobs

The OpenAI jobs check out the trusted base commit only. They fetch the PR ref
into Git object storage and generate `git diff` / changed-file text as data.
They execute `automation/reviewer.py` and `automation/REVIEW_RULES.md` from the
trusted base checkout.

### Claude patch-generation job

The Claude job checks out the trusted base commit, not the PR head. A separate
secret-free job converts a bounded set of PR files into text-only context.
Claude receives that text in its prompt.

The pinned Claude Code Action is invoked with:

- the workflow's read-only `GITHUB_TOKEN`, not a repository-write credential;
- `--bare`;
- `--permission-mode dontAsk`;
- `--tools ""`.

That removes built-in tools and denies unapproved MCP tool calls. Claude cannot
run tests, invoke shell commands, read arbitrary files, edit the workspace, or
push commits. Its only deliverable is a JSON-structured unified diff.

### Patch validation and apply jobs

Patch validation has no OpenAI or Claude secret. It rejects:

- empty or oversized patches;
- file renames;
- path traversal;
- protected automation paths;
- files outside the reviewer-derived allowlist;
- patches that do not apply cleanly to the exact reviewed head.

The apply job has repository write permission but no AI credential. It verifies
the PR branch has not moved, applies only the validated patch, creates one
commit, and pushes it with `GITHUB_TOKEN`. GitHub does not recursively trigger
new workflows from that token push, so the active run retains control.

### Test job

PR code is executed only in the test job. That job has no OpenAI key, no Claude
OAuth token, and only read-only repository permission.

## State machine

The controller state is persisted in a comment authored by
`github-actions[bot]`.

- `idle`: eligible for repair when all policy gates pass.
- `reserved`: one repair attempt has been consumed and is in progress.
- `completed`: the previous repair was applied, tested, and re-reviewed.
- `blocked`: a repair/validation/test/review stage failed or OpenAI returned
  `STOP`; automatic repair cannot resume by itself.
- `exhausted`: three PR-level automatic repair attempts have been consumed.

`blocked`, `exhausted`, and interrupted `reserved` states require the separate
manual reset workflow. The reset is auditable and records who reset it and the
previous state.

## Maximum repair count

Each repair is reserved before Claude is called. The reservation increments the
persistent PR-level count. A failed attempt remains consumed. Normal automation
cannot exceed three attempts across workflow runs. An explicit human reset may
authorize a new three-attempt budget and records that decision in reset history.
