#!/usr/bin/env python3
"""Requantize a target GGUF's matmul weights to ROCMFP types (AMD-native
formats with RDNA-tuned MMVQ/MMQ kernels in our ggml fork).

Policy (qwen35-family targets):
  Q4_K 2D weights (ffn_gate/up/down)          -> Q4_0_ROCMFP4_FAST
  Q8_0 2D weights (attn_*, ssm_*)             -> Q8_0_ROCMFPX
  Q6_K 2D weights (attn_output, output)       -> Q6_0_ROCMFPX
  token_embd, norms, 1D tensors, conv         -> unchanged

The from_float quantizers live in libggml (built with the rocmfpx types), so
this script requires --libggml (or auto-discovery under server/build-hip*).

Usage:
  python requant_target_rocmfp.py in.gguf out.gguf [--libggml path]
"""
import argparse
import concurrent.futures
import ctypes
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "llama.cpp" / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ROCMFP_TYPE_IDS = {
    "Q4_0_ROCMFP4":      100,
    "Q4_0_ROCMFP4_FAST": 101,
    "Q6_0_ROCMFPX":      102,
    "Q8_0_ROCMFPX":      103,
}

# source ggml type -> destination rocmfp type name
REQUANT_POLICY = {
    GGMLQuantizationType.Q4_K: "Q4_0_ROCMFP4_FAST",
    GGMLQuantizationType.Q8_0: "Q8_0_ROCMFPX",
    GGMLQuantizationType.Q6_K: "Q6_0_ROCMFPX",
}

KEEP_NAMES = {"token_embd.weight"}


class GgmlTypeTraits(ctypes.Structure):
    _fields_ = [
        ("type_name",            ctypes.c_char_p),
        ("blck_size",            ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size",            ctypes.c_size_t),
        ("is_quantized",         ctypes.c_bool),
        ("to_float",             ctypes.c_void_p),
        ("from_float_ref",       ctypes.c_void_p),
    ]


_GGML_FROM_FLOAT_T = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_void_p, ctypes.c_int64)


class GgmlLib:
    def __init__(self, path: str):
        self.path = path
        self.lib = ctypes.CDLL(path)
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(GgmlTypeTraits)
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_init.restype = None
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_blck_size.restype = ctypes.c_int64
        self.lib.ggml_blck_size.argtypes = [ctypes.c_int]
        self.lib.ggml_type_size.restype = ctypes.c_size_t
        self.lib.ggml_type_size.argtypes = [ctypes.c_int]
        self._from_float_cache: dict[int, object] = {}
        self._workers = max(1, int(os.environ.get("CONV_QUANT_THREADS",
                                                   os.cpu_count() or 8)))

    def blck_size(self, type_id: int) -> int:
        return int(self.lib.ggml_blck_size(type_id))

    def type_size(self, type_id: int) -> int:
        return int(self.lib.ggml_type_size(type_id))

    def row_size(self, type_id: int, n_per_row: int) -> int:
        return int(self.lib.ggml_row_size(type_id, n_per_row))

    def _from_float(self, type_id: int):
        fn = self._from_float_cache.get(type_id)
        if fn is None:
            self.lib.ggml_quantize_init(type_id)
            traits = self.lib.ggml_get_type_traits(type_id).contents
            if not traits.from_float_ref:
                raise RuntimeError(f"type {type_id} has no from_float_ref quantizer")
            fn = ctypes.cast(traits.from_float_ref, _GGML_FROM_FLOAT_T)
            self._from_float_cache[type_id] = fn
        return fn

    def quantize(self, type_id: int, arr_f32: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(arr_f32, dtype=np.float32)
        n_per_row = arr.shape[-1]
        nrows = arr.size // n_per_row
        blck = self.blck_size(type_id)
        if n_per_row % blck != 0:
            raise RuntimeError(f"n_per_row {n_per_row} not a multiple of blck_size "
                               f"{blck} for type {type_id}")
        row_bytes = self.row_size(type_id, n_per_row)
        total = row_bytes * nrows
        dst = (ctypes.c_char * total)()
        dst_addr = ctypes.addressof(dst)
        src_addr = arr.ctypes.data_as(ctypes.c_void_p).value
        fn = self._from_float(type_id)
        ELEM = 4
        workers = min(self._workers, nrows)

        def _quant_rows(r0: int):
            r1 = min(r0 + chunk_rows, nrows)
            nr = r1 - r0
            s = ctypes.cast(src_addr + r0 * n_per_row * ELEM, ctypes.POINTER(ctypes.c_float))
            d = ctypes.cast(dst_addr + r0 * row_bytes, ctypes.c_void_p)
            fn(s, d, ctypes.c_int64(nr * n_per_row))

        if workers <= 1:
            chunk_rows = nrows
            _quant_rows(0)
        else:
            chunk_rows = max(1, math.ceil(nrows / workers))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_quant_rows, range(0, nrows, chunk_rows)))
        buf = np.frombuffer(bytes(dst), dtype=np.uint8).copy()
        return buf.reshape((*arr.shape[:-1], row_bytes))


def find_libggml(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    root = Path(__file__).resolve().parent.parent
    for pat in ("build-hip*/**/libggml-base*.so", "build-hip*/**/libggml.so",
                "build*/**/libggml-base*.so"):
        hits = sorted(glob.glob(str(root / pat), recursive=True))
        if hits:
            return hits[0]
    return None


def register_rocmfp_type(type_id: int, lib: GgmlLib) -> None:
    bs = lib.blck_size(type_id)
    ts = lib.type_size(type_id)
    gguf.constants.GGML_QUANT_SIZES[type_id] = (bs, ts)
    try:
        gguf.quants.GGML_QUANT_SIZES[type_id] = (bs, ts)
    except Exception:
        pass


def copy_metadata(r: GGUFReader, w: GGUFWriter) -> None:
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    T = gguf.GGUFValueType
    for f in r.fields.values():
        if f.name in skip:
            continue
        ftype = f.types[0]
        val = f.parts[f.data[0]]
        if ftype == T.STRING:
            w.add_string(f.name, bytes(val).decode())
        elif ftype == T.ARRAY:
            sub = f.types[1]
            vals = [f.parts[i] for i in f.data]
            if sub == T.STRING:
                w.add_array(f.name, [bytes(v).decode() for v in vals])
            else:
                w.add_array(f.name, [np.asarray(v)[0].item() for v in vals])
        elif ftype == T.BOOL:
            w.add_bool(f.name, bool(val[0]))
        elif ftype == T.FLOAT32:
            w.add_float32(f.name, float(val[0]))
        elif ftype == T.FLOAT64:
            w.add_float64(f.name, float(val[0]))
        else:
            fn = {T.UINT32: w.add_uint32, T.INT32: w.add_int32,
                  T.UINT64: w.add_uint64, T.INT64: w.add_int64,
                  T.UINT8: w.add_uint8, T.INT8: w.add_int8,
                  T.UINT16: w.add_uint16, T.INT16: w.add_int16}[ftype]
            fn(f.name, val[0].item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--libggml", default=None)
    ap.add_argument("--skip-output", action="store_true",
                    help="keep output.weight (lm_head) at its source type")
    ap.add_argument("--only-q4k", action="store_true",
                    help="requantize only Q4_K tensors (FFN); keep Q8_0/Q6_K native")
    args = ap.parse_args()

    lib_path = find_libggml(args.libggml)
    if not lib_path:
        print("error: libggml not found; pass --libggml", file=sys.stderr)
        return 1
    lib = GgmlLib(lib_path)
    print(f"[info] libggml: {lib_path}")
    for tid in ROCMFP_TYPE_IDS.values():
        register_rocmfp_type(tid, lib)

    r = GGUFReader(args.input)
    arch = None
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[f.data[0]]).decode()
    if not arch:
        print("error: no general.architecture in input", file=sys.stderr)
        return 1

    w = GGUFWriter(args.output, arch)
    copy_metadata(r, w)

    n_q = n_keep = 0
    for t in r.tensors:
        shape = [int(x) for x in t.shape]  # ggml ne order
        dst_name = REQUANT_POLICY.get(t.tensor_type)
        keep = (
            dst_name is None or len(shape) != 2 or t.name in KEEP_NAMES or
            "norm" in t.name or shape[0] % 256 != 0 or
            (args.skip_output and t.name == "output.weight") or
            (args.only_q4k and t.tensor_type != GGMLQuantizationType.Q4_K)
        )
        if keep:
            w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
            n_keep += 1
            continue
        type_id = ROCMFP_TYPE_IDS[dst_name]
        f32 = dequantize(t.data, t.tensor_type).reshape(shape[::-1])
        buf = lib.quantize(type_id, f32)
        w.add_tensor(t.name, buf, raw_dtype=type_id)
        n_q += 1
        print(f"[requant] {t.name:36s} {t.tensor_type.name:5s} -> {dst_name} {tuple(shape)}")

    print(f"[info] writing {args.output} (requantized {n_q}, kept {n_keep})")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
