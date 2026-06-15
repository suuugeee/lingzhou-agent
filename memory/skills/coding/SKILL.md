---
name: coding
description: Use when Codex needs to implement, debug, refactor, or review code in a repository, especially tasks involving source files, failing tests, logs, runtime behavior, API wiring, or multi-file changes that require evidence-driven edits and verification.
---

# Coding

## Core Workflow

1. Locate the real repo root and inspect the current git state before editing.
2. Read the existing implementation path before proposing changes. Prefer `rg`, `rg --files`, focused `sed`, and targeted tests over broad exploration.
3. Identify the smallest code boundary that can solve the observed problem. Preserve local style, naming, contracts, and public behavior unless the task explicitly requires a broader redesign.
4. Make edits only after the execution path is understood. Use repo-native helpers, abstractions, fixtures, and test style.
5. Verify with the narrowest meaningful command first, then widen only when the touched surface is shared or risky.
6. Report the bug, changed files, verification commands, and remaining risk clearly.

## Evidence Rules

- Treat logs, test failures, stack traces, screenshots, and user-provided paths as primary evidence.
- If behavior is unclear, reproduce or inspect the exact path before coding.
- Do not infer a root cause from a symptom when a cheap command can verify it.
- When a task spans frontend, backend, config, database, or deployment, trace the full linked path before declaring completion.
- If the repo is dirty, protect unrelated user changes and avoid reverting files you did not intentionally modify.

## Editing Rules

- Keep diffs scoped to the problem and avoid unrelated cleanup.
- Prefer explicit code over clever abstractions unless an existing project pattern supports the abstraction.
- Add comments only where they prevent misreading of non-obvious logic.
- For Python, run compile or targeted tests after syntax-sensitive edits.
- For JavaScript/TypeScript, run the package's existing lint/typecheck/test command when available.
- For Java or Spring projects, follow existing module boundaries and centralized dependency management.

## Verification Ladder

Use this order unless the repo suggests a better one:

1. Syntax/import check for touched files.
2. Focused unit or module test for the changed behavior.
3. Integration or smoke command for the affected flow.
4. Full suite only when the change touches shared runtime, routing, persistence, or public contracts.

If a verification command cannot run, capture the exact reason and do not claim it passed.

## Loop Control

- Stop repeated low-information actions. After several `file.list`, `file.read`, `memory.search`, or similar probes, synthesize what is known and choose a higher-information action.
- If a task reaches completion, ensure stale task focus, temporary state, or working-memory anchors are cleared or explicitly justified.
- If model/provider/runtime failures interrupt coding, preserve the task state and next verification rather than converting the failure into a natural wait.

## Final Handoff

Include only high-signal information:

- Root cause or finding.
- Files changed.
- Tests or commands run.
- Any known limitation, skipped verification, or deployment step.
