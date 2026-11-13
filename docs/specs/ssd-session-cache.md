# One SSD session cache for every model

A design spec for extending the disk session cache (`server/src/server/kv/`,
described in `server/docs/PREFIX_CACHE.md`, "Disk Session Cache") so that
every model and every serving path keeps its sessions on the SSD, and so that
several `luce_server` processes on one machine share one cache directory with
one budget.

## 1. What exists

- The file format is model-agnostic: a snapshot's tensors are stored verbatim
  with the key tokens, an auxiliary blob and a checksum, in
  `<dir>/<layout>/<hash>.dkv`. A layout fingerprint (tensor names, types and
  row shapes) and an identity salt (model file, `--max-ctx`, chat template,
  KV types) keep one model from loading another's files.
- It reaches a model through `CheckpointStore`: `detach` hands a payload to
  the writer, `adopt` takes one read from disk. Today that store is the
  backend's snapshot slots.
- One process per directory: the first server takes an exclusive lock on
  `<dir>/.lock`; a second server on the same directory runs without a disk
  cache.

Where it is on: the single-sequence path of backends that export snapshots
(qwen3, qwen35 and its layer split, deepseek4 without paged attention, the
laguna layer split, the cluster head), and the qwen35 concurrent engine,
whose checkpoints sit in backend snapshot slots.

Where it is off:

- concurrent engines that keep their own checkpoint store: `qwen4exp` and
  deepseek4 paged;
- single-sequence backends that export nothing: `qwen4exp`, gemma4, laguna
  (not split), deepseek4 paged.

## 2. Engine-owned checkpoints

A concurrent engine that keeps its own checkpoints exposes them to the disk
cache through one small interface, `SeqEngine::prefix_store_payloads()`:

| Call | Meaning |
|------|---------|
| `tokens(id)` | positions checkpoint `id` holds; 0 when empty |
| `copy(id)` | a host copy for the writer; the checkpoint stays |
| `detach(id)` | hand the payload over as a host snapshot; the checkpoint is empty after |
| `take_displaced(id)` | the payload the last capture into `id` replaced, kept until claimed |
| `drop_displaced(id)` | free it without writing it |
| `adopt(id, snapshot)` | take a snapshot read from disk into checkpoint `id` |

The server wraps it as a third `CheckpointStore` (`EngineOwnedCheckpoints`),
beside the backend slots, so the disk reader and writer, the staging
checkpoint, the shutdown flush and the shared-prefix warm work the same for
every engine. The staging checkpoint keeps its id (the last slot).

The host snapshot is an `OwnedSnapshot` (moved to `common/owned_snapshot.h`
so `seq_engine.h` can name it). An engine packs its own metadata (position,
tokens, draft-layer state, logits) as extra tensors, so the file format does
not change and the layout fingerprint still separates layouts: variable
lengths sit in the row dimension, which the fingerprint ignores.

Engines:

- `qwen4exp` (concurrent and single-sequence): a snapshot in system memory
  is handed over without a copy, one on a GPU is copied to host. A disk read
  is taken in as is when the store is system memory. A capture keeps the
  checkpoint it replaces until the server claims it.
- deepseek4 paged: its snapshots already carry a metadata tensor; the
  engine's checkpoint tokens are packed beside it.
- gemma4 and laguna (single-sequence): `snapshot_ref` / `snapshot_adopt`
  over their snapshot tensors.

A backend whose snapshots need packing says so with
`ModelBackend::snapshot_exports()`, and implements `snapshot_copy`,
`snapshot_detach` and `snapshot_adopt`; the disk cache no longer treats a
missing `snapshot_ref` as memory-only for it.

## 3. One directory, several processes

- **Locks.** Each server holds a shared lock on `<dir>/.lock` for its
  lifetime. An older binary, which takes it exclusively, is refused while a
  new server runs, and an older binary already holding it keeps new servers
  off (as before). A short exclusive lock on `<dir>/.index.lock` covers every
  change to what is on disk: start-up cleanup, budget eviction, TTL expiry
  and publishing a write.
- **One view.** Under that lock a server syncs its index with the directory:
  it adds the files other servers wrote, drops the ones they deleted, and
  reads every file's `last_used` from its header. It then evicts the least
  recently used files of any model until the new file fits, and renames its
  file into place.
- **Writes and reads.** A write is a temporary file, `fdatasync`, rename,
  directory sync. Temporary names carry the writer's PID; start-up removes
  only those of processes that are gone. A file another server deletes while
  it is being read stays readable to the reader; one deleted before the open
  is a miss and leaves the index.
- **Budget and TTL.** They are the directory's: each server keeps the whole
  directory within its own `--kv-cache-budget` and `--kv-cache-ttl`, so with
  different values the smaller one wins. Queued writes that are not on disk
  yet count only for the server that queued them; the publish step re-checks
  the budget against the directory, so the directory exceeds it by at most
  the files being written at that moment.
- Windows keeps one process per directory.

## 4. Measurement

On one machine, two servers sharing one directory (a 4-slot `qwen4exp`
server and a single-sequence Qwen 3.8 27B server): for a long conversation
on each, the time to first token of a cold prefill, a RAM hit, and a restore
from the SSD after a RAM eviction and after a restart; the bytes written and
the write time; greedy output compared before and after an SSD restore; and a
small budget that makes one server evict the other's files.
