---
name: global-agent-rules
description: Global behavioral and engineering rules for coding agents. Use when writing, reviewing, or refactoring any code. Covers mindset, project structure and modularity, design, naming, errors, logging/tracing, security, testing evidence, scope/git safety, and commit messages.
license: MIT
---

# Global Agent Rules

Based on Karpathy's guidelines on LLM coding pitfalls, extended with structure, design, logging, evidence and delivery rules.

**Tradeoff:** These rules bias toward caution over speed. For trivial tasks, use judgment.

**Reading order:** general to specific. Mindset (A) governs everything below it. `§N` marks a cross-reference.

| Part | Sections | Question it answers |
|---|---|---|
| A. Mindset | §1-§4 | How do I approach the task? |
| B. Code | §5-§10 | How do I structure and write it? |
| C. Quality | §11 | How do I prove it works? |
| D. Delivery | §12-§13 | How do I ship it safely? |

---

## Priority Order

When rules conflict, the higher row wins. If the conflict is still unclear, stop and ask (§1).

| # | Priority | Sections |
|---|---|---|
| 1 | Explicit user instruction | - |
| 2 | Safety, scope, data protection | §10, §12 |
| 3 | Truthful evidence, no fabricated results | §11 |
| 4 | Surgical and simple changes | §2, §3 |
| 5 | Structure and dependency boundaries | §5 |
| 6 | Design, errors, logging | §6, §8, §9 |
| 7 | Naming and style | §7 |

Known conflicts:

- **Simplicity vs modularity:** don't add abstractions or layers just to look modular (§2), but never break dependency direction or module boundaries (§5).
- **Surgical vs structure:** apply §5 to **new** code only. Don't reorganize existing code to satisfy §5 unless the task asks for it (§3).
- **Logging vs surgical:** add logging/tracing at the boundaries of operations you create or change (§9). Don't add it to untouched code (§3).
- **Naming vs compatibility:** never rename persisted keys or public IDs to satisfy §7 without a tested migration (§10).

---

# Part A. Mindset

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Don't pick silently.
- If a simpler approach exists, say so (§2). Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- Before replacing working code, write down: old behavior, desired change, affected call sites, regression to preserve (§11).
- Read every producer and consumer of a field or method before changing its data shape.
- If a change forces a boundary violation (§5) or a scope breach (§12), stop and surface it. Don't work around it silently.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code. Add one only when it owns a real contract or removes demonstrated duplication.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios (but see §8 for real failures).
- If you write 200 lines and it could be 50, rewrite it.
- Don't keep old and new implementations in parallel as a safety blanket. Replace, verify (§11), delete the old one.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style and structure, even if you'd do it differently (§5).
- Unrelated dead code: mention it, don't delete it.
- Remove imports/variables/functions that YOUR changes made unused. Leave pre-existing dead code alone.
- Don't mass-rename existing symbols without evidenced need and a regression plan (§7, §11).

Test: every changed line traces directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

- "Add validation" -> write tests for invalid inputs, then make them pass.
- "Fix the bug" -> write a test that reproduces it, then make it pass.
- "Refactor X" -> ensure tests pass before and after.

For multi-step tasks, state a brief plan:

```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong criteria let you loop independently. Weak criteria ("make it work") force constant clarification. Report results per §11. Deliver per §12-§13.

---

# Part B. Code

## 5. Project Structure & Modularity

**Organize like a large project: feature first, layers inside, dependencies one-way.**

### Structure

- Top level is split by **feature/domain**, not by technical layer. One feature = one module that owns everything it needs.
- Inside a feature, split by **layer/role** only as far as responsibilities require (e.g. `api`, `domain`, `application`, `infrastructure`, `ui`, `tests`). Don't create empty layers up front (§2).
- Shared code lives in explicit, named shared modules (`core`, `shared-<purpose>`). Never in a catch-all `utils/` or `common/` (§7).
- Folder and file names describe a domain or responsibility (§7).
- One responsibility per file (§6). Group files by cohesion, not by file type.
- Tests mirror the structure of the code they cover and live with their feature (§11).
- Config, docs, scripts, and generated output each have their own dedicated location. Don't mix them into source folders.

### Module boundaries

- Every module exposes a small **public surface** (an entry point or `api` package). Everything else is internal and must not be imported from outside.
- Other modules depend on the public surface only. Never reach into another module's internals.
- A module owns its data and state. Others access them through its public surface, not by sharing storage or mutable objects (§10).
- Don't leak third-party or framework types across module boundaries unless that is the module's purpose.

### Dependency rules

- Dependencies point **one way**: `feature -> shared/core`, `outer layer -> inner layer`. Never the reverse.
- Inside a feature: `ui/api -> application -> domain`; `infrastructure -> domain` (implements its interfaces). `domain` imports nothing from other layers, frameworks, or I/O.
- **No circular dependencies**, between modules or between layers. If two modules need each other, extract the shared contract into a third module or invert the dependency with an interface.
- **Feature -> feature imports are forbidden by default.** Communicate through a shared contract, an explicit public API, or events. Document any exception (§12).
- `shared/core` must not depend on any feature.
- Entry points and composition roots (startup, wiring, DI) are the only places allowed to know about all modules.
- Adding a new dependency edge is a design decision: state it, justify it, check it against these rules.

### Working in an existing structure

- Follow the existing structure and conventions. Don't reorganize folders or move files unless the task requires it (§3).
- Place new code in the module that owns the responsibility. If none does, propose a new module before creating one (§1).
- When moving or splitting files, update every import and reference in the same change, and verify build and tests before and after (§11).

## 6. Function & Module Design

- One function, one responsibility, one reason to change. Split rendering, persistence, input handling, and execution.
- No hidden side effects: a getter, validator, renderer, or loader must not execute commands, write data, or hit the network.
- Separate verbs: register, resolve, validate, execute, render, persist (§7).
- No boolean-flag soup (`run(a, true, false, true)`). Use named options or separate functions.
- Make invalid states hard to construct: typed variants, explicit nullability, immutable data where practical. Avoid raw `Object`/`any` and unchecked casts.
- Validate untrusted input once, at its owning boundary (§5, §10). Don't re-parse it downstream.
- Event handlers must be idempotent, unregister properly, and never execute an action twice.
- Don't block UI/main/render threads with I/O or heavy work. Return state mutations to the owning thread.
- Public APIs define: arguments, ownership, lifecycle, failure result, threading. Record this in project docs, not in code comments, unless comments are requested.
- Avoid wrapper methods that add nothing (§2).

### Comments

- **No comments unless explicitly requested.** Express intent through names (§7), types, and structure. If code needs a comment to be understood, rename or restructure it.
- Never add commented-out code, history/changelog comments, or unowned TODOs.
- Existing comments: leave them untouched unless your change makes them wrong, then fix or remove only those (§3).

## 7. Naming

**Names describe stable purpose, never chronology.**

- Never use `V1`, `V2`, `New`, `Old`, `Legacy`, `Final`, `Fixed`, `Latest`, `Temp`, `Backup` or dates in file, folder, class, function, or config names. Git history tracks revisions.
- Use role names when two implementations genuinely coexist (`CurvedRenderer` vs `RadialRenderer`).
- Avoid vague buckets (`Misc`, `Helper`, `Util`, `Stuff`, `Common`, `Manager`, `Service`) when a domain-specific name exists (§5).
- Function names state the operation: `registerAction`, `resolveProfile`, `validateInput`. Avoid `process`, `handleData`, `doStuff` (§6).
- Booleans read as questions (`is`, `has`, `can`, `should`). Commands use verbs.
- Name magic values (`VISIBLE_ENTRY_COUNT = 7`). Don't confuse display limits with storage capacity.
- **Exceptions:** externally mandated names, schema migrations (`Migration002`), semantic API versions that coexist by contract, versioned release artifacts. Document why.
- Persisted keys and public IDs are compatibility contracts. Don't rename without a tested migration or alias (§10).

## 8. Error Handling

- Fail explicitly with actionable messages. No swallowed exceptions, no silent fallback, no generic `catch (Exception)` without a reason.
- Preserve the original cause when wrapping. Never replace it with a generic string.
- Handle each failure once, at the owning boundary (§5). Inner layers return structured errors or add debug breadcrumbs (§9).
- Expected invalid input -> rejection with a stable reason code. Unexpected failure -> error with full cause and trace ID (§9).
- Failures must not cause retry loops or surprise re-execution.

## 9. Logging & Tracing

**Every failure must be locatable by operation, module, function, and cause.**

- Create one trace ID per independent operation (user action, save, request, job). Never per frame, tick, loop iteration, or nested call.
- Pass the same trace ID through the whole chain, including async handoffs. Never let one operation's context leak into the next.
- Preserve the trace ID through exception wrapping (§8). Don't invent new IDs at each catch layer.
- Structured fields (snake_case): `trace_id`, `module` (match §5 module names), `operation`/`event`, `class`/`function`, `outcome`, `error_code`, `duration_ms` (when useful).
- Levels:
  - ERROR: unexpected failure, failed durable write, invariant violation. Full cause + trace ID. Once per failure.
  - WARN: non-fatal unexpected state. Deduplicate and rate-limit.
  - INFO: meaningful operation completion and lifecycle events.
  - DEBUG: decisions and branches. Off by default.
  - TRACE: opt-in noisy diagnostics only.
- Pure helpers and getters: no per-call logs. Diagnose them via the caller's trace and stack.
- High-frequency paths (render, tick, hover, tight loops): log state changes or rate-limited diagnostics only.
- Rate-limit identical failures: keep the first complete failure, count the repeats.
- Never log secrets, tokens, credentials, raw user content, personal identifiers, full addresses, or sensitive file contents (§10). Allowlist and mask.
- Logging must never crash the app, block hot paths, or alter execution order (§6).
- User-visible errors show a short error code + trace ID, without internals.
- Logging is verified like any other behavior (§11).

## 10. Persistence & Security

- Save with: validate -> write temp file -> atomic replace. Never delete the only valid copy before its replacement is safely written.
- Serialize stable IDs and data, not runtime object identities.
- Preserve unknown or future fields on read. Don't silently destroy user data.
- Define scope/profile precedence explicitly. Don't mutate across scopes.
- Classify sensitive actions (commands, deletes, external calls) and make them visible to the user.
- Prefer public, supported APIs for third-party integrations. Avoid undocumented internals and reflection.
- Keep secrets out of shared config files, logs (§9), and version control (§12).
- Preserve third-party license notices and provenance for any copied code.
- Validate untrusted input at its owning boundary (§6).

---

# Part C. Quality

## 11. Testing & Evidence

- Any behavior change needs a focused regression test covering normal, invalid, and missing-dependency paths, or an explicit NOT RUN / BLOCKED explanation (§4).
- Tests target contracts and outcomes, not implementation names. They live with their feature (§5).
- A passing unit test is not proof of real-world behavior.
- Report results as **PASS / FAIL / NOT RUN / BLOCKED / PARTIAL**, with the exact command, environment, and output.
- Static inspection is never a runtime PASS.
- No fabricated data, invented APIs, or success claims without execution evidence.
- Observability check on every new feature (§9): trace start/outcome, failure fields, no duplicate ERRORs, redaction, no per-call spam.
- Structure check on every change (§5): dependency direction respected, no cycles, no new feature-to-feature imports, imports updated after moves.
- Verify build and tests before and after refactors and file moves (§3).

---

# Part D. Delivery

## 12. Scope & Git Safety

- Edit only files within the task's scope (§3). Treat read-only references, vendored code, and upstream checkouts as untouchable.
- Preserve unrelated changes in the worktree. They belong to someone else.
- No commit, push, merge, or branch switch without explicit instruction.
- Never `git add -A` blindly. Inspect `git status` and `git diff`, then stage explicit paths only.
- Never stage secrets (§10) or generated output.
- Keep docs in sync with decisions and observed results (§11).
- A documented, justified exception is the only way to bypass a rule. State the constraint in the change.

## 13. Commit Messages

Applies only when committing is allowed (§12).

- Messages must be complete and self-explanatory.
- **Subject:** imperative mood, states what changed, ~72 chars max (e.g. `Fix duplicate execution on key release`).
- **Body:** WHY the change was made, WHAT changed (per module/file group when several, §5), and behavior before vs. after.
- Note affected call sites, breaking changes, migrations, and compatibility impact (§7, §10).
- State verification: exact test commands run and results (§11).
- Reference related issues, decisions, or trace/error codes (§9) when they exist.
- One logical change per commit (§3). Never mix unrelated changes, and never use vague messages like `fix`, `update`, `wip`, `misc changes`.
