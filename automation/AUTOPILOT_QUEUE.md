# Methodology Autopilot Queue

This document describes the Round 2 automation infrastructure that runs the
trusted M1 -> M2 -> M3 -> M4 methodology tasks sequentially and unattended,
built on top of the Round 1 OpenAI -> Claude repair loop
(`automation/AUTOFIX_SECURITY.md`, `.github/workflows/openai-autofix-round.yml`).

**This infrastructure never merges anything into `main` by itself.** The
queue's only output on `main` is one human-reviewed pull request.

## Entrypoint

`.github/workflows/methodology-autopilot.yml` is a `workflow_dispatch`-only
workflow. It:

1. Refuses to run unless it is dispatched with `github.ref ==
   'refs/heads/main'` and the `confirm` input is exactly `RUN`.
2. Creates a unique temporary integration branch,
   `autopilot/<run_id>/integration`, from the trusted main dispatch SHA.
3. Runs `M1`, `M2`, `M3`, `M4` strictly in sequence via the reusable
   `.github/workflows/autopilot-task.yml` workflow, each depending on the
   previous task's `status == 'merged'` output. Any task that does not reach
   a clean merged state stops the queue - later tasks are skipped.
4. Only after `M4` reports `status == 'merged'`, opens **one** cumulative PR
   from the integration branch to `main`. That PR may receive further OpenAI
   review/fix rounds (`cumulative_round_1/2/3` jobs) but this workflow
   contains no merge action for it anywhere. Merging into `main` is always a
   human decision.

## Per-task pipeline (`autopilot-task.yml`)

For each task (`M1`..`M4`), given the current integration branch:

1. `guard_integration` - hard-fails immediately if `integration_branch ==
   'main'`. Records the integration branch head SHA as the task's base.
2. `create_task_branch` - creates `autopilot/<run_id>/<task_id>` from that
   exact commit.
3. `prepare_context` - a **secret-free** job that reads
   `automation/tasks/<task_id>.json` and snapshots only the task's
   `context_files` as bounded plain text. Every declared context file is
   required: missing, binary, or over-limit context fails the task immediately
   instead of being silently omitted.
4. `claude_patch` - the **only** job with a Claude credential. It:
   - checks out the trusted base commit only (never untrusted PR content
     that doesn't exist yet at this point in the pipeline);
   - uses the pinned `anthropics/claude-code-action` commit
     `56cf60fde42f7b19c3abfd5c9c48b69a1288461f`;
   - has `contents: read` only, and no repository-write credential;
   - runs with `github_token: ${{ github.token }}`,
     `--permission-mode dontAsk`, `--tools ""`, `--output-format json`, and a
     structured JSON schema `{patch, summary}`;
   - returns only a unified diff and a summary; built-in Claude tools are
     disabled, so the model is not given shell/filesystem/GitHub mutation tools.

   This is credential separation plus tool suppression, not an OS/container
   sandbox around the third-party Claude Code action itself. The action runs
   normally on the GitHub-hosted runner and may make the network calls required
   for its OAuth/API operation. The security boundary is that this job has
   read-only repository permission and no repository-write credential; patch
   validation and repository writes happen later in separate jobs with no AI
   credentials.
5. `validate_patch` - a **secret-free** job that independently enforces the
   task's `allowed_files` allowlist, rejects `.github/` and `automation/`
   paths, rejects path traversal and renames, enforces a patch size limit,
   and requires the patch to `git apply --check` cleanly against the exact
   task branch head.
6. `apply_patch` - a **secret-free write** job (`contents: write`, no
   `OPENAI_API_KEY`, no `CLAUDE_CODE_OAUTH_TOKEN`) that re-verifies the
   branch has not moved, applies exactly the validated patch, and pushes one
   commit.
7. `test` - a **secret-free** job with no OpenAI key and no Claude OAuth
   token that compiles all Python and runs the full `pytest` suite. It
   preserves the `tests/pages` symlink workaround used by
   `openai-autofix-round.yml` for Streamlit `AppTest` page resolution.
8. `create_pr` - opens a PR whose **base is the temporary integration
   branch, never `main`**, with the full trusted task specification and
   acceptance criteria in the PR body, and creates a trusted
   `github-actions[bot]` auto-fix controller comment
   (`<!-- openai-autofix-controller:v3 -->`) with `autopilot_managed: true`.
9. `round_1` / `round_2` / `round_3` - call the existing reusable
   `openai-autofix-round.yml` with `trusted_automation: true`, exactly as
   `openai-reviewer.yml` does for human PRs, capped at 3 repair attempts.
10. `finalize_verdict` - resolves the verdict/head SHA from whichever round
    actually ran (or `STOP` if none did).
11. `merge_gate` - runs **only if the final verdict is `PASS`**. It
    hard-fails if `integration_branch == 'main'`, re-checks the integration
    branch has not drifted and the PR head has not moved since the PASS
    verdict, and only then squash-merges the task PR into the integration
    branch.
12. `report_status` - fails the whole reusable workflow call unless the
    merge actually completed with a `PASS` verdict, which is what stops the
    top-level queue on `STOP`, test failure, an unsafe/rejected patch,
    branch drift, or exhausting the 3-round repair budget without `PASS`.

## Trusted automation vs. normal human PRs

`openai-autofix-round.yml` gained one new optional input,
`trusted_automation` (default `false`). Normal calls from
`openai-reviewer.yml` never set it, so human PR trust rules
(`OWNER`/`MEMBER`/`COLLABORATOR`, same-repo) are completely unchanged. When
`trusted_automation: true`, the `reserve` job instead requires:

- the PR is same-repository (this check is never skipped);
- the PR is authored by `github-actions[bot]`;
- the PR head ref starts with `autopilot/`;
- the trusted controller comment's state contains `autopilot_managed: true`.

A normal (non-trusted) call can never operate on a controller state that has
`autopilot_managed: true`, and a `trusted_automation: true` call can never
operate on a controller state that lacks it - the two trust paths cannot be
mixed.

## Avoiding event recursion

Every queue PR contains an atomic hidden queue marker in its PR body at
creation time and uses an `autopilot/<run_id>/...` head ref. The standard
`pull_request_target` reviewer skips only when all of these are true: the PR
is same-repository, authored by `github-actions[bot]`, has a valid queue
marker whose run id matches the head ref, and is therefore identifiable before
the controller comment exists. A user-controlled branch prefix by itself is
never enough to bypass the normal reviewer.

## Stopping and resuming

The queue stops itself (no code needs to intervene) whenever:

- OpenAI returns `STOP`;
- the secret-free test job fails;
- Claude returns an empty patch or a patch that fails validation;
- the integration branch or a task PR's head drifts unexpectedly;
- the 3-round repair budget is exhausted without a `PASS` verdict.

In every one of these cases the affected task's reusable workflow call
fails, so the top-level `if: needs.mN.outputs.status == 'merged'` gate skips
every later task. The temporary integration branch and any task branches are
left in place for inspection; nothing is auto-merged. To resume investigate
the failed run, fix the underlying issue by hand (or use the existing
`openai-autofix-reset.yml` against the stuck task PR's controller comment if
it is `blocked`/`exhausted`/`reserved`), and re-dispatch a fresh queue run
once ready - it will always start a brand new integration branch from the
current `main`.

## Task specifications

`automation/tasks/M1.json` .. `M4.json` are the trusted task specs. Each
contains `task_id`, `title`, `depends_on`, `context_files` (read-only input
to Claude), `allowed_files` (the patch allowlist enforced by
`validate_patch`), `acceptance_criteria`, and `prohibited`. None of the four
specs list any path under `.github/` or `automation/` in `context_files` or
`allowed_files` - this is enforced both statically (see
`tests/test_autopilot_queue_contract.py`) and at runtime by
`prepare_context` and `validate_patch`.

## Scope of this Round 2 change

This infrastructure ships with **no M1-M4 application changes**. Running
`methodology-autopilot.yml` is what will actually execute the four tasks in
the future; it is a separate, explicit, human-triggered action.
