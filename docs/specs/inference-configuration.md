# Inference configuration plan

Status: draft

## Summary

Lucebox resolves inference settings in several places today:

- `server_main.cpp` parses CLI flags and applies server policy.
- `BackendArgs` carries backend construction requests.
- `prepare_backend()` inspects the GGUF and checks compatibility.
- `create_backend()` copies a subset of `BackendArgs` into an
  architecture-specific config.
- Backends read operational and experimental environment variables.
- `ServerConfig` owns HTTP, request, cache, and generation behavior.
- Model-card sidecars provide model-authored generation defaults.

These parts have different owners and lifetimes. They should not become one
application-wide config object. The first change should instead finish the
backend boundary that already exists:

```text
BackendArgs -> BackendPlan -> create_backend(plan)
```

`BackendPlan` is the only value accepted by backend construction. It contains
the exact request that passed model inspection and compatibility checks.
`ServerConfig` and model-card data stay separate.

Later changes can add a config file, an argument registry, and runtime
introspection on top of this boundary.

## Goals

- Give backend construction one validated input.
- Make a raw `BackendArgs` and stale plan impossible to pair.
- Preserve the current CLI, environment variables, defaults, diagnostics, and
  runtime behavior in the first change.
- Keep HTTP and generation settings out of the backend plan.
- Keep model cards limited to model-authored generation guidance.
- Leave a clear place for values discovered during backend initialization.
- Allow a later config file to populate the same request types as the CLI.

## Non-goals

The first change does not:

- add YAML or JSON configuration;
- replace every environment variable;
- group every CLI flag into new public types;
- move `ServerConfig` into the backend layer;
- select kernels or memory budgets from model cards;
- require a per-architecture public plan variant;
- change inference behavior.

## Current flow

The native server parses backend and server settings together in
`server/src/server/server_main.cpp`:

```text
CLI
 |
 +-> BackendArgs
 |
 +-> ServerConfig
 |
 +-> temporary CLI state and environment variables
```

`prepare_backend()` then inspects `general.architecture`, resolves placement,
checks `model_capabilities.h`, and applies `feature_gate.cpp`. It returns a
`ResolvedBackendPlan`, but that plan does not contain the `BackendArgs` that
passed those checks.

Construction receives both values:

```cpp
create_backend(const BackendArgs & args,
               const ResolvedBackendPlan & plan);
```

The server also changes DDTree and SpecLA fields after preparation. The
factory therefore compares the raw arguments with the plan and runs the gate
again. This catches some drift at runtime, but the API still permits the
invalid pair.

The configuration sources also have different resolution times:

- GGUF architecture and static placement are available before construction.
- Free VRAM, allocated KV capacity, loaded checkpoint geometry, and selected
  kernels may be known only during backend initialization.
- Model-card defaults are resolved after model identity is known and affect
  generation policy, not backend construction.
- Per-request sampling and reasoning options are resolved by `HttpServer`.

A useful design must preserve these times instead of claiming that every
value can be resolved before initialization.

## Target ownership

Use the existing names where possible. The separation matters more than new
terminology.

### `BackendArgs`

`BackendArgs` remains the request passed into backend preparation. The CLI,
an embedding caller, or a future config-file loader may populate it.

It is not an effective configuration. Defaults and sentinel values still need
resolution, and the model may not support the requested combination.

Future work may replace ambiguous sentinel values with `std::optional` or an
explicit `auto` state. That work is not required to seal the construction
boundary.

### `BackendPlan`

`BackendPlan` is an immutable, backend-owned construction plan. Only
`prepare_backend()` can create it.

It owns:

- the effective `BackendArgs` snapshot used by construction;
- owned target and draft paths;
- inspected GGUF model facts;
- the selected compiled and target backends;
- warnings and backend admission decisions needed by construction.

The plan does not own `ServerConfig`, model cards, request defaults, CORS,
HTTP ports, disk cache paths, or upstream credentials.

The plan must own its model and draft path strings. Copying the current
`const char *` fields without their storage would leave the supposedly sealed
plan dependent on the caller's lifetime.

### `ServerConfig`

`ServerConfig` keeps its existing HTTP and request responsibilities. Backend
features may participate in admission, but that does not make the full server
config a backend dependency.

The server should receive only the backend facts that it needs for startup
reporting and request behavior. It should not pass its full config into the
factory.

### `BackendInfo`

`BackendInfo` is a later reporting type for values observed during
initialization. Examples include:

- actual device memory;
- allocated paged-KV and KVFlash capacity;
- effective KV data types;
- loaded model geometry;
- selected kernels;
- runtime fallbacks or degradations.

`BackendInfo` is not an input to construction. The backend produces it after
initialization. The startup banner and `/props` can eventually combine a
redacted view of `BackendPlan`, `BackendInfo`, and `ServerConfig`.

## First change: seal backend construction

The first implementation should make the backend plan the only authoritative
construction input.

### Public flow

```cpp
BackendPreparation prepared = prepare_backend(args, features);
if (!prepared.ok()) {
    // Preserve the current error and exit-status mapping.
}

for (const std::string & warning : prepared.warnings) {
    // Preserve the current warning text and order.
}

auto backend = create_backend(prepared.plan);
```

The factory API becomes:

```cpp
std::unique_ptr<ModelBackend> create_backend(const BackendPlan & plan);
```

The current two-argument overload is removed:

```cpp
create_backend(const BackendArgs &, const ResolvedBackendPlan &);
```

If out-of-tree callers need the one-argument convenience API, it may remain as
a wrapper:

```cpp
std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args) {
    BackendPreparation prepared = prepare_backend(args);
    if (!prepared.ok()) {
        return nullptr;
    }
    return create_backend(prepared.plan);
}
```

The wrapper must not contain a second dispatch path.

### Plan construction

`prepare_backend()` should take or copy the request, own both path strings,
inspect the model, resolve the narrow backend-specific adjustments that occur
after preparation today, and validate the final request.

The existing SpecLA behavior is the important case:

- supported single-device Qwen enables DDTree;
- KVFlash disables SpecLA verification but keeps ordinary DDTree;
- unsupported architecture or placement disables SpecLA;
- a missing required draft remains an error;
- an operator-supplied DDTree tau keeps its current precedence.

This normalization should be a private helper inside backend preparation. A
public `prepare`, `patch`, and `finalize` protocol would make each caller learn
the ordering problem that the factory should hide.

`prepare_backend()` may run the pure feature gate before and after the narrow
normalization when needed. The first check preserves current error precedence.
The second check validates the exact request stored in the plan.

### Construction

`create_backend(const BackendPlan &)` reads only the plan's private argument
snapshot and inspected facts. The existing architecture dispatch may continue
to build concrete configs such as `Qwen35Config`, `LagunaBackendArgs`, and
`DeepSeek4BackendConfig` internally.

The first change does not need to expose those configs through a public
`std::variant`. A public variant would pull every backend header into the
common factory contract. The plan can adopt internal variants later if they
remove enough factory duplication to justify the dependency cost.

## Model cards remain generation-only

Model-card sidecars record recommendations from the model publisher. They are
the right place for:

- sampling defaults;
- output limits;
- reasoning effort tiers;
- thinking markers;
- model-specific generation guidance.

They are not the right place for:

- GPU placement;
- paged attention;
- KV data types;
- chunk sizes;
- speculative implementation selection;
- expert residency;
- kernel selection.

Those decisions depend on the compiled backend, hardware topology, available
memory, quantization, and deployment goal. Lucebox owns them, not the model
publisher.

The existing `share/model_cards/` format and resolution order remain
unchanged.

## Later changes

### Central argument registry

The native server currently defines help text, parsing, validation, aliases,
and destinations in a long manual branch. Move that metadata into one argument
registry after the backend boundary is sealed.

Each public option should define:

- CLI spellings;
- an optional environment alias;
- value type and parser;
- help text;
- destination in `BackendArgs` or `ServerConfig`;
- whether the option is stable, experimental, or deprecated.

This follows the useful part of llama.cpp's common argument registry without
requiring Lucebox to copy its full parameter object.

### Config file

Add a config file only after CLI and backend preparation share typed inputs.
A file loader should populate `BackendArgs` and `ServerConfig`; it should not
introduce a parallel runtime config.

An initial file could use this shape:

```yaml
model: /models/qwen3.8-27b.gguf

placement:
  target: auto
  draft: auto

serving:
  context: auto
  concurrency: 8

speculation:
  mode: auto
  draft_model: /models/qwen3.8-dflash2.gguf

cache:
  kv_type: auto
  paged: auto
```

Suggested runtime precedence is:

```text
CLI > environment alias > config file > automatic policy > built-in default
```

Generation defaults use a separate order:

```text
request > server override > model card > family fallback > hard fallback
```

Keeping the orders separate prevents an engine profile from silently
overwriting model-authored sampling guidance.

### Environment variables

The environment-variable reference already classifies debug, kill-switch,
test, and removal-candidate variables. Keep those categories.

Migrate documented operational settings only when their owning typed config
exists. Debug instrumentation, benchmark switches, build flags, and emergency
kill switches may remain environment variables. A project to eliminate every
environment variable would add risk without simplifying normal deployment.

### Runtime report

Add `BackendInfo` after backends expose enough actual values to make it useful.
Do not estimate free-memory-derived values in `BackendPlan`.

The report used by `/props` must be a stable, redacted projection. It must not
serialize upstream keys, private internal paths, or implementation-only debug
state.

## Verification

The first implementation should prove the boundary directly:

1. Build `dflash_server` and the common backend targets.
2. Confirm that no call site uses `create_backend(args, plan)`.
3. Assert that `BackendPlan` cannot be default-constructed or mutated by a
   caller.
4. Prepare a plan from temporary path storage and confirm that the plan owns
   the paths.
5. Cover supported SpecLA, SpecLA with KVFlash, unsupported SpecLA, an explicit
   tau, and a missing draft without loading GPU weights.
6. Run `test_feature_gate` unchanged.
7. Compare server diagnostics and exit statuses before and after the change.
8. Run representative server smoke launches when model fixtures are available.

The config-file phase should add precedence tests before it adds deployment
profiles.

## Alternatives rejected

### One application-wide config

A single config would make serialization convenient, but every consumer would
see unrelated backend, HTTP, generation, and debug state. It would recreate the
same coupling as a flat argument object with more nesting.

### Put runtime profiles in model cards

This mixes two authors and two release cycles. Model publishers define
generation recommendations. Lucebox defines engine policy for a particular
binary and machine.

### Add YAML first

A config file would improve launch commands but leave the current split
between raw arguments, a partial plan, environment state, and backend defaults.
Settle the backend boundary before adding another input source.

### Move every environment variable into the plan

Many variables exist for profiling, tests, kernel experiments, and temporary
kill switches. Treating them as stable launch configuration would expand the
normal interface and make experimental implementation details permanent.

## Related designs

- [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py)
  converts broad engine arguments into grouped model, cache, scheduler,
  parallel, load, and compilation configs before engine construction.
- [SGLang](https://github.com/sgl-project/sglang/issues/19139) has an RFC to
  split its large `ServerArgs` object because changing the broad object has
  caused unexpected side effects.
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html)
  accepts nested configuration files and gives explicit CLI flags precedence
  over file values.
- [TGI](https://huggingface.co/docs/text-generation-inference/en/reference/launcher)
  derives checkpoint and memory-dependent defaults when the operator omits
  them.
- [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.h)
  centralizes option declarations and supports model presets while keeping its
  parser as the boundary.

These engines still expose many controls. Their useful shared property is that
input parsing, effective engine configuration, and per-request generation are
distinct stages.

## Open questions

- Does the project need to preserve `create_backend(const BackendArgs &)` for
  out-of-tree callers during the first implementation?
- Should the final plan keep the current `BackendFeatureConfig` name, or would
  `BackendAdmissionContext` describe its role more accurately?
- Which operational environment variables are stable enough to join the
  future argument registry?
- Should the config file support named profiles in its first version, or only
  explicit nested values?
