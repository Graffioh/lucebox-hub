"""Python mirror of the OFlash capture-ring format.

Single source of truth: server/src/common/oflash/oflash_format.h. Any change
there bumps RING_VERSION and must be mirrored here; the golden-bytes tests on
both sides (server/test/test_oflash_unit.cpp ring_step_record_golden_bytes and
tests/test_ring_format.py) keep the two honest.

The ring is a single-producer (engine decode thread) / single-consumer (this
trainer) byte ring in a POSIX shm segment (/dev/shm/<name>). `head`/`tail`
are monotonic byte counts; a record at logical offset L lives at
data_offset + (L % capacity). Records never wrap mid-record: a PAD record
covers the buffer tail instead.
"""

from __future__ import annotations

import mmap
import struct
import time
from dataclasses import dataclass, field

import numpy as np

RING_MAGIC = 0x4F464C31  # "OFL1"
RING_VERSION = 2
HEADER_SIZE = 256
RECORD_HEADER_SIZE = 32

# Header field offsets (bytes) — see oflash_format.h layout comment.
OFF_MAGIC = 0
OFF_VERSION = 4
OFF_CAPACITY = 8
OFF_DATA_OFFSET = 16
OFF_HEAD = 64
OFF_TAIL = 128
OFF_DROPPED_RECORDS = 192
OFF_DROPPED_BYTES = 200
OFF_DRAFTER_HASH = 208
OFF_N_CAPTURE = 216
OFF_HIDDEN = 220
OFF_BLOCK_SIZE = 224
OFF_TOPK = 228
OFF_VOCAB = 232
OFF_TARGET_HASH = 240
OFF_DRAFTER_SEMANTICS_HASH = 248

REC_PAD = 0
REC_CONTEXT = 1
REC_STEP = 2
REC_SEQ_END = 3


@dataclass
class RingInfo:
    capacity: int
    data_offset: int
    drafter_hash: int
    target_hash: int
    drafter_semantics_hash: int
    n_capture_layers: int
    hidden: int
    block_size: int
    topk: int
    vocab: int

    @property
    def row_elems(self) -> int:
        return self.n_capture_layers * self.hidden


@dataclass
class Record:
    type: int
    seq_id: int
    pos: int
    n_rows: int
    t_mono_ns: int
    # CONTEXT/STEP: bf16 feature rows as uint16 bits, [n_rows, row_elems].
    feat_u16: np.ndarray | None = None
    # STEP labels.
    n_labels: int = 0
    draft_tok: np.ndarray | None = None    # i32 [n_labels]
    target_tok: np.ndarray | None = None   # i32 [n_labels]
    accept_flags: np.ndarray | None = None  # u8 [n_labels]
    accept_len: int = 0
    bonus_tok: int = -1
    topk_ids: np.ndarray | None = None     # i32 [n_labels, K]
    topk_lp: np.ndarray | None = None      # f32 [n_labels, K]


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Convert an array of uint16 bf16 bit patterns to float32."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


class RingReader:
    """Attach to the engine's shm ring and iterate records.

    The engine owns the segment; open the backing file read-write (the tail
    cursor is ours to advance) but never touch producer fields.
    """

    def __init__(self, path: str):
        # path: "/dev/shm/<name>" or a plain file (tests).
        self._f = open(path, "r+b")
        self._mm = mmap.mmap(self._f.fileno(), 0)
        deadline = time.monotonic() + 5.0
        while self._u32(OFF_MAGIC) != RING_MAGIC:
            if time.monotonic() > deadline:
                raise RuntimeError(f"ring at {path} never became ready")
            time.sleep(0.05)
        version = self._u32(OFF_VERSION)
        if version != RING_VERSION:
            raise RuntimeError(
                f"ring version {version} != supported {RING_VERSION}")
        self.info = RingInfo(
            capacity=self._u64(OFF_CAPACITY),
            data_offset=self._u64(OFF_DATA_OFFSET),
            drafter_hash=self._u64(OFF_DRAFTER_HASH),
            target_hash=self._u64(OFF_TARGET_HASH),
            drafter_semantics_hash=self._u64(OFF_DRAFTER_SEMANTICS_HASH),
            n_capture_layers=self._u32(OFF_N_CAPTURE),
            hidden=self._u32(OFF_HIDDEN),
            block_size=self._u32(OFF_BLOCK_SIZE),
            topk=self._u32(OFF_TOPK),
            vocab=self._u32(OFF_VOCAB),
        )

    def close(self) -> None:
        self._mm.close()
        self._f.close()

    # ── raw accessors ────────────────────────────────────────────────
    def _u32(self, off: int) -> int:
        return struct.unpack_from("<I", self._mm, off)[0]

    def _u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self._mm, off)[0]

    @property
    def head(self) -> int:
        return self._u64(OFF_HEAD)

    @property
    def tail(self) -> int:
        return self._u64(OFF_TAIL)

    @tail.setter
    def tail(self, value: int) -> None:
        struct.pack_into("<Q", self._mm, OFF_TAIL, value)

    @property
    def dropped_records(self) -> int:
        return self._u64(OFF_DROPPED_RECORDS)

    def backlog(self) -> int:
        return self.head - self.tail

    def _read_span(self, logical: int, n: int) -> bytes:
        """Contiguous read; records never wrap so one slice suffices."""
        off = self.info.data_offset + (logical % self.info.capacity)
        return self._mm[off:off + n]

    # ── record iteration ─────────────────────────────────────────────
    def read_next(self) -> Record | None:
        """Parse the record at tail and advance it. None when caught up.

        PAD records are consumed transparently.
        """
        while True:
            if self.head - self.tail < 8:
                return None
            at = self.tail
            # A buffer-tail gap can be smaller than a full record header;
            # PADs there carry only their first 8 bytes (type + size).
            to_end = self.info.capacity - (at % self.info.capacity)
            if to_end < RECORD_HEADER_SIZE:
                rtype, size = struct.unpack("<II", self._read_span(at, 8))
                if rtype != REC_PAD or size != to_end:
                    raise RuntimeError(f"corrupt tail-gap record at {at}")
                self.tail = at + size
                continue
            if self.head - self.tail < RECORD_HEADER_SIZE:
                return None
            hdr = self._read_span(at, RECORD_HEADER_SIZE)
            rtype, size, seq_id, pos, n_rows, t_ns = struct.unpack(
                "<IIQiiQ", hdr)
            if (size < RECORD_HEADER_SIZE or size % 8 != 0
                    or size > to_end):
                raise RuntimeError(f"corrupt record size {size} at {at}")
            if rtype == REC_PAD:
                self.tail = at + size
                continue
            if rtype not in (REC_CONTEXT, REC_STEP, REC_SEQ_END):
                raise RuntimeError(f"unknown record type {rtype} at {at}")
            if n_rows < 0:
                raise RuntimeError(f"negative row count {n_rows} at {at}")
            if self.head - at < size:
                return None  # producer mid-publish; retry later
            body = self._read_span(at + RECORD_HEADER_SIZE,
                                   size - RECORD_HEADER_SIZE)
            try:
                rec = self._parse(rtype, seq_id, pos, n_rows, t_ns, body)
            except (ValueError, struct.error, OverflowError, IndexError) as e:
                raise RuntimeError(f"corrupt record body at {at}: {e}") from e
            self.tail = at + size
            return rec

    def _parse(self, rtype: int, seq_id: int, pos: int, n_rows: int,
               t_ns: int, body: bytes) -> Record:
        rec = Record(type=rtype, seq_id=seq_id, pos=pos, n_rows=n_rows,
                     t_mono_ns=t_ns)
        if rtype == REC_SEQ_END:
            return rec
        row = self.info.row_elems
        feat_bytes = n_rows * row * 2
        rec.feat_u16 = np.frombuffer(
            body, dtype=np.uint16, count=n_rows * row).reshape(n_rows, row).copy()
        if rtype == REC_CONTEXT:
            return rec
        off = feat_bytes
        n_labels, topk_k = struct.unpack_from("<ii", body, off)
        off += 8
        rec.n_labels = n_labels
        if n_labels > 0:
            rec.draft_tok = np.frombuffer(
                body, dtype=np.int32, count=n_labels, offset=off).copy()
            off += n_labels * 4
            rec.target_tok = np.frombuffer(
                body, dtype=np.int32, count=n_labels, offset=off).copy()
            off += n_labels * 4
            rec.accept_flags = np.frombuffer(
                body, dtype=np.uint8, count=n_labels, offset=off).copy()
            off += (n_labels + 7) & ~7
        rec.accept_len, rec.bonus_tok = struct.unpack_from("<ii", body, off)
        off += 8
        if topk_k > 0 and n_labels > 0:
            rec.topk_ids = np.frombuffer(
                body, dtype=np.int32, count=n_labels * topk_k,
                offset=off).reshape(n_labels, topk_k).copy()
            off += n_labels * topk_k * 4
            rec.topk_lp = np.frombuffer(
                body, dtype=np.float32, count=n_labels * topk_k,
                offset=off).reshape(n_labels, topk_k).copy()
        return rec


# ── Test-only writer (mirrors OFlashRing::push2) ─────────────────────

@dataclass
class RingWriter:
    """File-backed ring writer for tests; NOT used in production (the
    engine is the only producer)."""

    path: str
    capacity: int
    n_capture_layers: int = 1
    hidden: int = 2
    block_size: int = 2
    topk: int = 0
    vocab: int = 10
    drafter_hash: int = 0
    target_hash: int = 0
    drafter_semantics_hash: int = 0
    _head: int = field(default=0, init=False)

    def __post_init__(self):
        with open(self.path, "wb") as f:
            f.truncate(HEADER_SIZE + self.capacity)
        self._f = open(self.path, "r+b")
        self._mm = mmap.mmap(self._f.fileno(), 0)
        struct.pack_into("<I", self._mm, OFF_VERSION, RING_VERSION)
        struct.pack_into("<Q", self._mm, OFF_CAPACITY, self.capacity)
        struct.pack_into("<Q", self._mm, OFF_DATA_OFFSET, HEADER_SIZE)
        struct.pack_into("<Q", self._mm, OFF_DRAFTER_HASH, self.drafter_hash)
        struct.pack_into("<Q", self._mm, OFF_TARGET_HASH, self.target_hash)
        struct.pack_into("<Q", self._mm, OFF_DRAFTER_SEMANTICS_HASH,
                         self.drafter_semantics_hash)
        struct.pack_into("<I", self._mm, OFF_N_CAPTURE, self.n_capture_layers)
        struct.pack_into("<I", self._mm, OFF_HIDDEN, self.hidden)
        struct.pack_into("<I", self._mm, OFF_BLOCK_SIZE, self.block_size)
        struct.pack_into("<I", self._mm, OFF_TOPK, self.topk)
        struct.pack_into("<I", self._mm, OFF_VOCAB, self.vocab)
        struct.pack_into("<I", self._mm, OFF_MAGIC, RING_MAGIC)

    def push(self, rtype: int, seq_id: int, pos: int, n_rows: int,
             payload: bytes, t_mono_ns: int = 0) -> None:
        size = (RECORD_HEADER_SIZE + len(payload) + 7) & ~7
        to_end = self.capacity - (self._head % self.capacity)
        if size > to_end:
            pad = struct.pack("<IIQiiQ", REC_PAD, to_end, 0, 0, 0, 0)
            self._write_at(self._head, pad[:min(to_end, len(pad))])
            self._head += to_end
        rec = struct.pack("<IIQiiQ", rtype, size, seq_id, pos, n_rows,
                          t_mono_ns) + payload
        rec += b"\x00" * (size - len(rec))
        self._write_at(self._head, rec)
        self._head += size
        struct.pack_into("<Q", self._mm, OFF_HEAD, self._head)

    def _write_at(self, logical: int, data: bytes) -> None:
        off = HEADER_SIZE + (logical % self.capacity)
        self._mm[off:off + len(data)] = data

    def close(self):
        self._mm.close()
        self._f.close()


def pack_step_labels(draft_tok, target_tok, accept_flags, accept_len,
                     bonus_tok, topk_ids=None, topk_lp=None) -> bytes:
    """Build the STEP label section (after the feat block)."""
    n = len(draft_tok)
    k = 0 if topk_ids is None else int(np.asarray(topk_ids).shape[-1])
    out = struct.pack("<ii", n, k)
    out += np.asarray(draft_tok, dtype=np.int32).tobytes()
    out += np.asarray(target_tok, dtype=np.int32).tobytes()
    flags = np.asarray(accept_flags, dtype=np.uint8).tobytes()
    out += flags + b"\x00" * (((n + 7) & ~7) - n)
    out += struct.pack("<ii", accept_len, bonus_tok)
    if k:
        out += np.asarray(topk_ids, dtype=np.int32).tobytes()
        out += np.asarray(topk_lp, dtype=np.float32).tobytes()
    return out
