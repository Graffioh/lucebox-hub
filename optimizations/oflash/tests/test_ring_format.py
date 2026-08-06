"""Contract tests for the capture-ring mirror (oflash.ring_format).

The binding layout is server/src/common/oflash/oflash_format.h; the C++ side
pins the same bytes in server/test/test_oflash_unit.cpp. The golden STEP
record below is copied verbatim from TEST_CASE ring_step_record_golden_bytes
there — if either side changes, both tests must change in the same commit.
"""

from __future__ import annotations

import struct

import numpy as np

from oflash.ring_format import (
    HEADER_SIZE,
    OFF_HEAD,
    REC_CONTEXT,
    REC_PAD,
    REC_SEQ_END,
    REC_STEP,
    RECORD_HEADER_SIZE,
    RingReader,
    RingWriter,
    bf16_bits_to_f32,
    pack_step_labels,
)


def f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """Truncating f32 -> bf16 bits. Tests use bf16-exact values only, so
    truncation vs round-to-nearest cannot differ."""
    return (np.asarray(arr, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_bf16_bits_to_f32_known_values():
    bits = np.array([0x3F80, 0x4000, 0xBFC0, 0x0000], dtype=np.uint16)
    np.testing.assert_array_equal(bf16_bits_to_f32(bits),
                                  np.array([1.0, 2.0, -1.5, 0.0], dtype=np.float32))


def test_roundtrip_context_step_seq_end(tmp_path):
    path = str(tmp_path / "ring")
    w = RingWriter(path, capacity=1 << 16, n_capture_layers=2, hidden=4,
                   block_size=4, topk=3, vocab=100, drafter_hash=0x1122334455667788)
    row = 2 * 4

    # CONTEXT: bf16-exact feature values so the bit conversion is lossless.
    ctx_vals = (np.arange(3 * row, dtype=np.float32).reshape(3, row) * 0.25) - 1.5
    ctx_bits = f32_to_bf16_bits(ctx_vals)
    w.push(REC_CONTEXT, seq_id=7, pos=100, n_rows=3,
           payload=ctx_bits.tobytes(), t_mono_ns=11)

    # STEP: n_rows > n_labels (committed bonus row carries features, no label).
    rng = np.random.default_rng(0)
    step_bits = rng.integers(0, 1 << 16, size=(5, row), dtype=np.uint16)
    draft = np.array([3, 7, 11, 15], dtype=np.int32)
    target = np.array([3, 7, 2, 9], dtype=np.int32)
    flags = np.array([1, 1, 0, 0], dtype=np.uint8)
    topk_ids = rng.integers(0, 100, size=(4, 3)).astype(np.int32)
    topk_lp = rng.standard_normal((4, 3)).astype(np.float32)
    labels = pack_step_labels(draft, target, flags, accept_len=3, bonus_tok=2,
                              topk_ids=topk_ids, topk_lp=topk_lp)
    w.push(REC_STEP, seq_id=7, pos=103, n_rows=5,
           payload=step_bits.tobytes() + labels, t_mono_ns=22)

    w.push(REC_SEQ_END, seq_id=7, pos=108, n_rows=0, payload=b"", t_mono_ns=33)

    r = RingReader(path)
    assert r.info.capacity == 1 << 16
    assert r.info.n_capture_layers == 2
    assert r.info.hidden == 4
    assert r.info.block_size == 4
    assert r.info.topk == 3
    assert r.info.vocab == 100
    assert r.info.drafter_hash == 0x1122334455667788
    assert r.info.row_elems == row

    rec = r.read_next()
    assert rec is not None
    assert rec.type == REC_CONTEXT
    assert (rec.seq_id, rec.pos, rec.n_rows, rec.t_mono_ns) == (7, 100, 3, 11)
    np.testing.assert_array_equal(rec.feat_u16, ctx_bits)
    np.testing.assert_array_equal(bf16_bits_to_f32(rec.feat_u16), ctx_vals)
    assert rec.n_labels == 0 and rec.draft_tok is None

    rec = r.read_next()
    assert rec is not None
    assert rec.type == REC_STEP
    assert (rec.seq_id, rec.pos, rec.n_rows, rec.t_mono_ns) == (7, 103, 5, 22)
    np.testing.assert_array_equal(rec.feat_u16, step_bits)
    assert rec.n_labels == 4
    np.testing.assert_array_equal(rec.draft_tok, draft)
    np.testing.assert_array_equal(rec.target_tok, target)
    np.testing.assert_array_equal(rec.accept_flags, flags)
    assert rec.accept_len == 3
    assert rec.bonus_tok == 2
    np.testing.assert_array_equal(rec.topk_ids, topk_ids)
    np.testing.assert_array_equal(rec.topk_lp, topk_lp)

    rec = r.read_next()
    assert rec is not None
    assert rec.type == REC_SEQ_END
    assert (rec.seq_id, rec.pos, rec.n_rows, rec.t_mono_ns) == (7, 108, 0, 33)
    assert rec.feat_u16 is None

    assert r.read_next() is None
    assert r.backlog() == 0
    r.close()
    w.close()


def test_pack_step_labels_roundtrip_odd_labels_no_topk(tmp_path):
    """STEP with n_rows=0 (budget-clamped) and odd n_labels exercises the
    accept_flags pad-to-8 path with no feature block in front."""
    path = str(tmp_path / "ring")
    w = RingWriter(path, capacity=1 << 12)
    draft = [5, 6, 7]
    target = [5, 8, 1]
    flags = [1, 0, 0]
    labels = pack_step_labels(draft, target, flags, accept_len=1, bonus_tok=8)
    # 8 (counts) + 12 + 12 + 8 (flags padded) + 8 (accept_len, bonus) = 48.
    assert len(labels) == 48
    w.push(REC_STEP, seq_id=2, pos=0, n_rows=0, payload=labels)

    r = RingReader(path)
    rec = r.read_next()
    assert rec is not None and rec.type == REC_STEP
    assert rec.n_rows == 0
    assert rec.feat_u16.shape == (0, r.info.row_elems)
    assert rec.n_labels == 3
    np.testing.assert_array_equal(rec.draft_tok, np.array(draft, dtype=np.int32))
    np.testing.assert_array_equal(rec.target_tok, np.array(target, dtype=np.int32))
    np.testing.assert_array_equal(rec.accept_flags, np.array(flags, dtype=np.uint8))
    assert rec.accept_len == 1
    assert rec.bonus_tok == 8
    assert rec.topk_ids is None and rec.topk_lp is None
    r.close()
    w.close()


def test_wrap_emits_pad_and_reader_skips_it(tmp_path):
    path = str(tmp_path / "ring")
    w = RingWriter(path, capacity=256)  # n_capture=1, hidden=2 -> 4 B/row
    rec_a_rows = 30  # 32 hdr + 120 feat = 152 bytes
    rec_b_rows = 20  # 32 hdr + 80 feat = 112 bytes; 104 left -> PAD + wrap
    feat_a = np.arange(rec_a_rows * 2, dtype=np.uint16).reshape(rec_a_rows, 2)
    feat_b = np.arange(rec_b_rows * 2, dtype=np.uint16).reshape(rec_b_rows, 2) + 1000
    size_a = RECORD_HEADER_SIZE + feat_a.nbytes
    size_b = RECORD_HEADER_SIZE + feat_b.nbytes
    pad_size = 256 - size_a

    w.push(REC_CONTEXT, seq_id=1, pos=0, n_rows=rec_a_rows, payload=feat_a.tobytes())
    # Consume A before the wrapping push (the test-only writer has no
    # drop-on-full; wrapped bytes reuse the consumed region, like the engine).
    r = RingReader(path)
    rec = r.read_next()
    assert rec is not None and rec.pos == 0
    np.testing.assert_array_equal(rec.feat_u16, feat_a)
    assert r.tail == size_a

    w.push(REC_CONTEXT, seq_id=1, pos=rec_a_rows, n_rows=rec_b_rows,
           payload=feat_b.tobytes())

    # The writer physically emitted a PAD record covering the buffer tail.
    with open(path, "rb") as f:
        raw = f.read()
    pad_type, pad_len = struct.unpack_from("<II", raw, HEADER_SIZE + size_a)
    assert pad_type == REC_PAD
    assert pad_len == pad_size
    head = struct.unpack_from("<Q", raw, OFF_HEAD)[0]
    assert head == size_a + pad_size + size_b  # record B starts at logical 256

    # The reader consumes the PAD transparently: next record is B.
    rec = r.read_next()
    assert rec is not None and rec.pos == rec_a_rows
    np.testing.assert_array_equal(rec.feat_u16, feat_b)
    assert r.read_next() is None
    assert r.tail == head
    r.close()
    w.close()


# Copied verbatim from server/test/test_oflash_unit.cpp
# TEST_CASE(OFlashUnitFixture, ring_step_record_golden_bytes).
KGOLDEN = bytes((
    0x02, 0, 0, 0, 64, 0, 0, 0,        # type=STEP, size=64
    1, 0, 0, 0, 0, 0, 0, 0,            # seq_id=1
    5, 0, 0, 0, 1, 0, 0, 0,            # pos=5, n_rows=1
    0, 0, 0, 0, 0, 0, 0, 0,            # t_mono_ns=0
    0x80, 0x3F, 0x00, 0x40,            # feat bf16 bits [1.0, 2.0]
    2, 0, 0, 0, 0, 0, 0, 0,            # n_labels=2, topk_k=0
    7, 0, 0, 0, 9, 0, 0, 0,            # draft_tok
    11, 0, 0, 0, 13, 0, 0, 0,          # target_tok
    0, 0, 0, 0,                        # pad to 64
))


def test_step_record_golden_bytes(tmp_path):
    """Byte-level fixture shared with the C++ writer.

    Ring dims and record contents match the C++ test exactly (n_capture=1,
    hidden=2, block=2, topk=0, vocab=10). NOTE: the label blob is the raw
    24 bytes the C++ test pushed — it deliberately omits accept_flags /
    accept_len / bonus, so this record is NOT a parseable STEP; the assertion
    is on raw serialized bytes only, which is the actual cross-language
    contract (framing, padding, field order, endianness).
    """
    assert len(KGOLDEN) == 64
    path = str(tmp_path / "ring")
    w = RingWriter(path, capacity=1 << 20, drafter_hash=0x0102030405060708)
    feat = struct.pack("<HH", 0x3F80, 0x4000)
    labels = struct.pack("<6i", 2, 0, 7, 9, 11, 13)
    w.push(REC_STEP, seq_id=1, pos=5, n_rows=1, payload=feat + labels, t_mono_ns=0)
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE + 64)
    assert struct.unpack_from("<Q", raw, OFF_HEAD)[0] == 64
    assert raw[HEADER_SIZE:HEADER_SIZE + 64] == KGOLDEN
    w.close()
