# Agent memory for Lucebox clients: plan

A plan for giving coding clients that run against `luce_server` a
persistent, per-repository memory learned from past sessions. Two
tracks:

- **A.** Run Letta Code, which ships such a memory, as one more client
  in `harness/clients/`, and measure whether its memory helps a local
  model.
- **C.** A harness-agnostic memory layer: transcripts captured once,
  reflected offline, and delivered as plain files that every client
  already reads.

A comes first. It tests the idea with an implementation as shipped. C
follows only if A's measurement passes.

## 1. Background: how Letta Code builds memory

The reference is [letta-ai/letta-code](https://github.com/letta-ai/letta-code)
(Apache-2.0), read at commit `6c8b08f` (2026-10-22). The relevant
sources are:

- `src/agent/subagents/builtin/init-v2.md`
- `src/agent/subagents/builtin/reflection-v2.md`
- `src/skills/builtin/maintaining-core-memory/SKILL.md`
- `src/skills/builtin/maintaining-deferred-memory/SKILL.md`

The design is also described in Letta's posts *Context Repositories*
(2026-02), *Skill Learning* (2025-12) and *Sleep-time Compute*
(2025-04).

**Layout.** Memory is a git repository of Markdown files:

| Part | Where | Loaded |
|------|-------|--------|
| Core memory | root `*.md` and root `MEMORY.md` | compiled into the system prompt every turn (65,536 chars total, 20,000 per file by default) |
| Deferred memory | subdirectories, each with its own `MEMORY.md` | on demand, guided by one-line entries in the root index and by file frontmatter `description`s |
| Skills | `skills/` | procedures with a trigger, loaded like agent skills |

Every change is a commit. Concurrent memory subagents work in git
worktrees and merge back.

**Formation.**

1. *Init:* reads `AGENTS.md`/`CLAUDE.md`, the package manifest and the
   README, and writes a 5–8 file skeleton. Its rules are "index, not
   encyclopedia" and "generalize, don't memorize".
2. *Reflection* (background, "sleep-time"): triggered by step count, by
   a compaction event, or manually. It reads recent transcripts, and
   may replay already-reflected ones to deduplicate and resolve
   contradictions.
   - It looks for, in this order: mistakes and corrections, then
     preferences and patterns, then new facts, then contradictions,
     then reusable procedures (which become skills).
   - It skips one-off task state, line numbers, exact errors,
     temporary paths and anything cheap to rederive.
   - It edits core memory as a revision of a prompt: rules that share
     a cause merge into one principle with its reason, and a
     contrastive example is added only when a principle keeps failing.
3. *Defragmentation:* reorganizes memory into 15–25 focused files.

**Evidence.** On Terminal-Bench 2.0, skills learned from trajectories
gave +9 points absolute. Adding the verifier's test logs to reflection
gave +15.7 points. Feedback on failures matters.

**Local use.** Letta Code has a `local` backend, so no Letta Cloud is
needed. `letta connect openai-compatible --base-url <url>` targets any
OpenAI-compatible server, and the local model path is built on pi's
model layer (`src/backend/pi-*`).

## 2. Goals and non-goals

Goals:

- Measure, on a real repository with graded tasks, whether memory
  written by the locally served model improves its work: success rate,
  turns, tokens and tool calls.
- If it does, ship one memory layer that works for every supported
  client (`pi`, `omp`, `hermes`, `codex`, `claude_code`, `opencode`,
  …) without per-client re-implementation.

Non-goals:

- Changing any client's own memory features.
- Hand-tuning memory content for a particular repository. Every rule
  in the layer must be repository- and language-independent.

## 3. Track A: Letta Code as a client

### A1. Launcher

- Add a `letta` entry to `harness/client_test_runner.py`
  (`install="npm"`, package `@letta-ai/letta-code`, version pinned).
- Add `harness/clients/run_letta.sh`, following `run_pi.sh`: it sources
  `common.sh`, isolates `HOME` under `$LOG_DIR`, selects the `local`
  backend, connects to `$BASE_URL/v1` as `openai-compatible`, and runs
  headless with `PROMPT_FILE`.
- Memory and settings live under `$LOG_DIR`. A run never touches the
  user's real `~/.letta`.

### A2. Compatibility smoke test (CPU or a small model)

- Run tool calls over `chat/completions` and over `responses`.
- Check streaming, and the context-window and max-token reporting.
- Run `/init` on a small repo and check that the memory repository is
  created and committed.
- Trigger one manual reflection and check it commits a memory change.
- Record any server incompatibility as a separate issue or PR against
  `luce_server`, not a workaround in the launcher.

### A3. Measurement: does memory help?

Graded tasks from one repository, each with tests that decide pass or
fail, split by commit time into a **train** half and an **exam** half.
Reflection only ever sees train-half transcripts, so exam transcripts
never reach memory.

| Arm | Memory |
|-----|--------|
| M0 | none (fresh agent, init disabled) |
| M1 | `/init` only |
| M2 | `/init` plus reflection over train transcripts |
| M3 | M2, with verifier test logs added to the reflected transcripts |

- **Metrics on the exam half:** pass rate, turns, total tokens and tool
  calls per success.
- **Same everywhere:** model, sampling settings, context window and
  number of seeds.
- **Pass rule (A-gate):** M2 or M3 beats M0 on pass rate, or matches it
  with clearly fewer turns or tokens. The intervals must be wide enough
  to support the read. If the exam is too small to resolve the
  expected effect, the result is reported as inconclusive and the exam
  grows before any decision.
- **Memory hygiene:** for every committed memory version, record its
  size and the share of tokens that are identifiers or paths existing
  in the repository. That share tracks whether memory holds principles
  or copies of the code.

### A4. Output

A short report with the arms, metrics, intervals and memory statistics,
plus three sample memory diffs read by hand.

## 4. Track C: harness-agnostic memory layer

This track starts only after A-gate passes. It keeps Letta's memory
format and prompts, and replaces only the orchestration.

### C1. Transcript capture

- A pass-through proxy between client and server, on the pattern of
  `harness/clients/session_inject_proxy.py`. It records each
  request/response pair to JSONL per session, tagged with the
  repository and commit.
- Capture is opt-in and local only.
- Known secret patterns are scrubbed before anything is written.
- Done at the proxy, capture covers every client with one
  implementation. A server-side option can replace the proxy later.

### C2. Reflection job

- Letta's prompts are vendored pristine, with license and `NOTICE`, at
  a pinned commit under `harness/third_party/letta_code/`. Our changes
  are recorded as marked commits on top.
- The job is a small runner that drives the vendored reflection prompt
  against `luce_server` with a minimal tool set (`bash` and `edit`,
  confined to the memory directory).
- It runs on a schedule (nightly, or after N new sessions) and commits
  to a per-repository memory repository.
- The memory size limits are enforced by a pre-commit check, as Letta
  does.
- A transcript enters reflection only after its session ends. Graded
  sessions include their test logs (A3's M3).

### C3. Delivery to clients

The memory is rendered into files clients already load:

- core memory becomes a fenced, generated block in the repository's
  `AGENTS.md`, or a client-specific equivalent;
- skills go into the client's skills directory;
- deferred memory is copied as files, linked from the generated block.

For each client in `harness/clients/`, record whether it loads
`AGENTS.md` and skills, and how. This is verified per client, not
assumed (for example, `run_omp.sh` currently passes `--no-skills`). A
client that loads neither gets the core block through its
system-prompt option.

### C4. Re-check

Re-run A3's M0 and M2 arms through `pi` and `omp` with C-delivered
memory, to confirm that A's result was not specific to the Letta
harness.

## 5. Risks

- **Reflection quality with a local model.** Letta reports that general
  models write memories that are generic or too specific. The model
  writing memory is the weakest link, and A3 measures it directly.
- **Exam size.** Graded tasks are scarce per repository, and small
  exams cannot resolve small effects (A3 pass rule).
- **Staleness.** Memory can describe code that later changes. Principle
  over path (the hygiene metric) limits this. Repository-referencing
  items should name the commit they were learned at.
- **Feedback loops.** The model writes the memory it later reads, so
  errors can compound. Every memory version is a git commit, and the
  A3 exam can be re-run on any version to roll back.
- **Privacy.** Transcripts can hold secrets and personal data. Capture
  is opt-in, scrubbed and local.

## 6. Milestones

| # | Step | Needs |
|---|------|-------|
| 1 | A1 launcher + A2 smoke test | CPU or small model |
| 2 | Graded task set with train/exam split | existing task tooling |
| 3 | A3 arms M0–M3 | GPU time |
| 4 | A-gate decision | report |
| 5 | C1 capture proxy, C2 reflection job, C3 delivery | after A-gate |
| 6 | C4 re-check on `pi` and `omp` | GPU time |

## 7. Open questions

- Should reflection run on the same model that serves the client, or
  may a stronger offline model write memory?
- Is memory per repository, or per user and repository, when several
  people share a repository?
- Generated `AGENTS.md` block versus a separate file the client is
  pointed at: which do maintainers accept in their repositories?
