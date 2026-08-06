"""CPU-only state-management regressions for the OFlash trainer."""

from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np

from oflash.ring_format import REC_CONTEXT, Record
from oflash.trainer import SeqStore, StepSample, Trainer


class _Control:
    def __init__(self):
        self.commands = queue.Queue()
        self.logs: list[str] = []

    def send_event(self, event: str, **fields) -> None:
        pass

    def log(self, message: str) -> None:
        self.logs.append(message)


def _trainer() -> Trainer:
    cfg = SimpleNamespace(
        start_generation=0,
        reservoir_rows=128,
        train_ctx=64,
    )
    ring = SimpleNamespace(info=SimpleNamespace(block_size=8))
    return Trainer(cfg, ring, _Control())


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
            pos=10) -> StepSample:
    return StepSample(
        pos=pos,
        draft_tok=np.array([1, 2], dtype=np.int32),
        target_tok=np.array([2, 3], dtype=np.int32),
        accept_flags=np.array(flags, dtype=np.uint8),
        accept_len=2,
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
