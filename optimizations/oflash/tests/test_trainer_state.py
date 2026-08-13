"""CPU-only state-management regressions for the OFlash trainer."""

from __future__ import annotations

import queue
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import oflash.trainer as trainer_module
from oflash.identity import fnv1a64
from oflash.ring_format import REC_CONTEXT, Record
from oflash.trainer import (
    SeqStore,
    StepSample,
    Trainer,
    _max_adapter_generation,
)


class _Control:
    def __init__(self):
        self.commands = queue.Queue()
        self.logs: list[str] = []
        self.events: list[tuple[str, dict]] = []

    def send_event(self, event: str, **fields) -> None:
        self.events.append((event, fields))

    def log(self, message: str) -> None:
        self.logs.append(message)


def _trainer(*, out_dir="/tmp/oflash-test-does-not-exist",
             device="cpu") -> Trainer:
    cfg = SimpleNamespace(
        start_generation=0,
        out_dir=out_dir,
        requested_device=device,
        dtype="auto",
        drafter_sha256="a" * 64,
        target_sha256="b" * 64,
        drafter_semantics=(
            "v1;rope=49742400;swa=2048;pattern=10;mask=248070"),
        profile="default",
        rank=4,
        alpha=32.0,
        reservoir_rows=128,
        train_ctx=64,
        seed=0,
    )
    ring = SimpleNamespace(info=SimpleNamespace(
        block_size=8,
        drafter_hash=int("a" * 16, 16),
        target_hash=int("b" * 16, 16),
        drafter_semantics_hash=fnv1a64(cfg.drafter_semantics),
    ))
    return Trainer(cfg, ring, _Control())


def test_ring_identity_binds_both_models_and_resolved_semantics():
    trainer = _trainer()
    assert trainer._validate_ring_identity()

    trainer._ring.info.target_hash ^= 1
    assert not trainer._validate_ring_identity()

    trainer = _trainer()
    trainer._ring.info.drafter_semantics_hash ^= 1
    assert not trainer._validate_ring_identity()


def test_newer_sequence_retires_older_live_store_without_seq_end():
    trainer = _trainer()
    old = SeqStore(seq_id=1, first_pos=0, max_pos=1)
    old.feat[0] = np.zeros(2, dtype=np.uint16)
    trainer._seqs[1] = old

    trainer._ingest(Record(
        type=REC_CONTEXT,
        seq_id=2,
        pos=0,
        n_rows=1,
        t_mono_ns=0,
        feat_u16=np.zeros((1, 2), dtype=np.uint16),
    ))

    assert 1 not in trainer._seqs
    assert old.evicted  # no trainable step: released rather than leaked
    assert 2 in trainer._seqs


def _sample(*, flags=(1, 1), bonus=-1, block_size=2,
            accept_len=None, pos=10) -> StepSample:
    n = len(flags)
    return StepSample(
        pos=pos,
        draft_tok=np.arange(1, n + 1, dtype=np.int32),
        target_tok=np.arange(2, n + 2, dtype=np.int32),
        accept_flags=np.array(flags, dtype=np.uint8),
        accept_len=n if accept_len is None else accept_len,
        bonus_tok=bonus,
        topk_ids=None,
        topk_lp=None,
        block_size=block_size,
    )


def test_tree_rejection_signal_includes_bonus_or_short_spine():
    assert not _sample().has_reject
    assert _sample(flags=(1, 0)).has_reject
    assert _sample(bonus=7).has_reject
    assert _sample(block_size=8).has_reject


def test_training_stops_after_first_rejection_conditioning_boundary():
    root_reject = _sample(flags=(1, 0, 0, 0), accept_len=1,
                          block_size=4)
    assert root_reject.valid_train_rows(q_len=4) == 1

    second_reject = _sample(flags=(1, 1, 0, 0), accept_len=2,
                            block_size=4)
    assert second_reject.valid_train_rows(q_len=4) == 2

    all_accepted = _sample(flags=(1, 1, 1, 1), accept_len=4,
                           block_size=4)
    assert all_accepted.valid_train_rows(q_len=4) == 3


def test_fresh_samples_over_microbatch_cap_are_drained_not_dropped():
    trainer = _trainer()
    trainer._micro_max = 1
    store = SeqStore(seq_id=1, first_pos=0, max_pos=64)
    for pos in range(64):
        store.feat[pos] = np.zeros(2, dtype=np.uint16)
    first = _sample(flags=(1, 0), pos=64)
    second = _sample(flags=(1, 0), pos=64)
    trainer._fresh = [(store, first), (store, second)]
    trainer._fresh_rows = first.n_labels + second.n_labels

    batch1 = trainer._build_batch()
    assert [item[1] for item in batch1] == [first]
    assert trainer._fresh == [(store, second)]
    assert trainer._fresh_rows == second.n_labels

    batch2 = trainer._build_batch()
    assert [item[1] for item in batch2] == [second]
    assert trainer._fresh == []
    assert trainer._fresh_rows == 0


def test_reservoir_replay_sampling_is_seeded_and_reproducible():
    left = _trainer()
    right = _trainer()
    for seq_id in range(4):
        store = SeqStore(seq_id=seq_id, first_pos=0, max_pos=64)
        for pos in range(64):
            store.feat[pos] = np.zeros(2, dtype=np.uint16)
        store.steps = [_sample(flags=(1, 0), pos=64),
                       _sample(flags=(1, 1, 0), pos=64)]
        left._reservoir.append(store)
        right._reservoir.append(store)

    left._micro_max = 8
    right._micro_max = 8
    left_draw = [(store.seq_id, sample.n_labels)
                 for store, sample, _start in left._build_batch()]
    right_draw = [(store.seq_id, sample.n_labels)
                  for store, sample, _start in right._build_batch()]

    assert left_draw == right_draw
    assert len(left_draw) == 8


def test_generation_resumes_above_every_existing_adapter(tmp_path):
    for name in ("adapter-gen2.safetensors", "adapter-gen11.safetensors",
                 "adapter-genX.safetensors", "unrelated"):
        (tmp_path / name).touch()
    assert _max_adapter_generation(str(tmp_path)) == 11
    assert _trainer(out_dir=str(tmp_path))._generation == 11


def test_oom_reduces_context_then_releases_training_memory():
    trainer = _trainer()
    trainer._train_ctx = 128
    trainer._optimizer = SimpleNamespace(
        zero_grad=lambda **_kwargs: None)
    trainer._mirror = SimpleNamespace()
    trainer._build_batch = lambda: [object()]

    def oom(_batch):
        raise RuntimeError("HIP out of memory")

    trainer._optimize = oom
    trainer._train_step()
    assert trainer._enabled
    assert trainer._train_ctx == 64

    trainer._train_step()
    assert not trainer._enabled
    assert trainer._optimizer is None
    assert trainer._mirror is None
    assert any("minimum 64-row" in line for line in trainer._control.logs)


def test_requested_gpu_never_silently_falls_back_to_cpu(monkeypatch):
    trainer = _trainer(device="1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer._validate_ring_identity = lambda: True
    trainer._drain_only_loop = lambda: 0

    assert trainer.run() == 0
    assert trainer._device == "cuda"
    assert trainer._dtype == torch.float16
    assert not trainer._enabled
    assert any("refusing unsafe CPU fallback" in line
               for line in trainer._control.logs)


def test_poisoned_accelerator_error_disables_training():
    trainer = _trainer()
    trainer._optimizer = SimpleNamespace(
        zero_grad=lambda **_kwargs: None)
    trainer._mirror = SimpleNamespace()
    trainer._build_batch = lambda: [object()]
    trainer._optimize = lambda _batch: (_ for _ in ()).throw(
        RuntimeError("HIP error: an illegal memory access was encountered"))

    trainer._train_step()

    assert not trainer._enabled
    assert trainer._optimizer is None
    assert trainer._mirror is None
    assert any("accelerator context is unsafe" in line
               for line in trainer._control.logs)


def test_non_oom_runtime_error_is_skipped_without_disabling():
    trainer = _trainer()
    trainer._optimizer = SimpleNamespace(
        zero_grad=lambda **_kwargs: None)
    trainer._mirror = SimpleNamespace()
    trainer._build_batch = lambda: [object()]
    trainer._optimize = lambda _batch: (_ for _ in ()).throw(
        RuntimeError("synthetic recoverable error"))

    trainer._train_step()

    assert trainer._enabled
    assert trainer._mirror is not None
    assert any("train step failed" in line for line in trainer._control.logs)


def test_rollback_ack_is_emitted_after_trainer_state_is_restored():
    trainer = _trainer()
    timeline = []
    trainer._rollback = lambda gen: timeline.append(("rollback", gen))
    trainer._control.send_event = (
        lambda event, **fields: timeline.append((event, fields["generation"])))
    trainer._control.commands.put(("rollback", 17))

    assert trainer._drain_commands() is None
    assert timeline == [("rollback", 17), ("rollback_ack", 17)]


def test_rollback_revalidates_promoted_adapter_identity(
        tmp_path, monkeypatch):
    trainer = _trainer(out_dir=str(tmp_path))
    promoted = tmp_path / "adapter-gen3.safetensors"
    promoted.touch()
    trainer._promoted_path = str(promoted)
    trainer._promoted_generation = 3
    trainer._drafter_semantics = "v1;rope=49742400;swa=2048;pattern=10;mask=248070"
    loaded = []
    trainer._mirror = SimpleNamespace(
        load_lora_state_numpy=lambda state: loaded.append(state),
        reset_lora=lambda: loaded.append("reset"),
        lora_parameters=lambda: [],
    )
    checked = []

    def load_validated(*args, **kwargs):
        checked.append((args, kwargs))
        return {"ok": 1}

    monkeypatch.setattr(
        trainer_module, "load_validated_adapter", load_validated)
    trainer._make_optimizer = lambda: None

    trainer._rollback(9)

    assert checked == [(
        (str(promoted), "a" * 64, "b" * 64,
         trainer._drafter_semantics, 4, 32.0),
        {"generation": 3, "profile": "default"},
    )]
    assert loaded == [{"ok": 1}]


def test_self_disable_emits_engine_visible_state_event():
    trainer = _trainer()

    trainer._disable("synthetic accelerator fault", release_model=False)

    assert not trainer._enabled
    assert trainer._control.events == [
        ("training_disabled", {"reason": "synthetic accelerator fault"})]


def test_load_oom_enters_drain_only_without_respawn_retry():
    trainer = _trainer()
    trainer._validate_ring_identity = lambda: True
    trainer._preflight_accelerator = lambda: None
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        raise MemoryError("synthetic host OOM")

    trainer._load_mirror = load
    trainer._drain_only_loop = lambda: 0

    assert trainer.run() == 0
    assert calls == 1
    assert not trainer._enabled
    assert trainer._mirror is None
    assert trainer._optimizer is None
    assert any("capture-only mode" in line for line in trainer._control.logs)


@pytest.mark.parametrize("failure_phase", ("preflight", "mirror"))
def test_failure_traceback_is_collectable_before_drain_only(failure_phase):
    trainer = _trainer()
    trainer._validate_ring_identity = lambda: True
    allocation_ref = None
    error_ref = None

    class Allocation:
        pass

    class CollectableError(RuntimeError):
        __slots__ = ("__weakref__",)

    def fail():
        nonlocal allocation_ref, error_ref
        allocation = Allocation()
        error = CollectableError("synthetic out of memory")
        allocation_ref = weakref.ref(allocation)
        error_ref = weakref.ref(error)
        raise error

    def should_not_run():
        raise AssertionError("mirror load continued after preflight failure")

    if failure_phase == "preflight":
        trainer._preflight_accelerator = fail
        trainer._load_mirror = should_not_run
    else:
        trainer._preflight_accelerator = lambda: None
        trainer._load_mirror = fail

    def drain():
        # run() must have left the except scope before entering this long-lived
        # loop and performed its post-exception collection, otherwise the
        # exception traceback retains both objects.
        assert allocation_ref is not None and allocation_ref() is None
        assert error_ref is not None and error_ref() is None
        return 0

    trainer._drain_only_loop = drain

    assert trainer.run() == 0
    assert not trainer._enabled


def test_auto_dtype_uses_fp32_cpu_and_fp16_gpu():
    cpu = _trainer(device="cpu")
    gpu = _trainer(device="1")
    assert cpu._dtype == torch.float32
    assert gpu._dtype == torch.float16
