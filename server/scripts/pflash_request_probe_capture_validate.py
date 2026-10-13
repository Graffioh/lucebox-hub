#!/usr/bin/env python3
"""A6 capture validation + CAPTURE.json manifest.
  validate.py OUT_DIR --probe GGUF [--recapture OTHER_DIR] [--trunk-convs 20]
(a) every stored row and logit finite; (b) the request trunk (numpy, float64, exact GELU) on the stored f32 rows
reproduces the runtime's own logits within 1e-5 on the first --trunk-convs conversations (0 = all); (c) with --recapture, the
re-captured conversations are byte-identical (rows, logits, index entries); (d) coverage of labelled messages and the
missing list; (e) storage. Writes OUT_DIR/CAPTURE.json; exit 0 only if all checks pass."""
import argparse, json, math, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps/llama.cpp/gguf-py"))
import gguf

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--recapture", default="")
ap.add_argument("--probe", required=True, help="the request-probe GGUF the capture ran with")
ap.add_argument("--trunk-convs", type=int, default=20)
a = ap.parse_args()
out = Path(a.out)

r = gguf.GGUFReader(a.probe)
T = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
eps = float(r.fields["requestprobe.norm_eps"].parts[-1][0])
nw, nb, w1, b1, w2, b2 = [T["requestprobe." + n] for n in ("norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")]
verf = np.vectorize(math.erf)
def trunk(x):
    x = x.astype(np.float64); m = x.mean(1, keepdims=True); v = x.var(1, keepdims=True)
    h = ((x - m) / np.sqrt(v + eps) * nw + nb) @ w1.T + b1
    return (0.5 * h * (1 + verf(h / math.sqrt(2)))) @ w2.reshape(-1) + b2

def load(d):
    convs = {}
    for idx in sorted(Path(d).glob("*.index.jsonl")):
        base = str(idx)[: -len(".index.jsonl")]
        rows = np.memmap(base + ".rows.f32", "<f4", "r")
        logits = np.fromfile(base + ".logits.f32", "<f4")
        for line in open(idx):
            e = json.loads(line)
            h = e["hidden"]; o = e["row_offset"]; n = e["rows"]
            convs[e["conversation_id"]] = (e, rows[o * h:(o + n) * h].reshape(n, h), logits[o:o + n])
    return convs

parts = [json.loads(p.read_text()) for p in sorted(out.glob("part*of*.json"))]
convs = load(out)
report = {"parts": [p["part"] for p in parts], "conversations": len(convs)}
# (a)
bad = [cid for cid, (e, rows, lg) in convs.items() if not np.isfinite(rows).all() or not np.isfinite(lg).all()]
report["a_finite"] = {"pass": not bad, "nonfinite_conversations": bad}
# (b)
gaps, n = [], 0
want = len(convs) if a.trunk_convs <= 0 else min(a.trunk_convs, len(convs))
for cid, (e, rows, lg) in list(convs.items())[:want]:
    n += 1
    gaps.append(float(np.abs(trunk(np.asarray(rows)) - lg).max()) if rows.size else 0.0)
report["b_trunk"] = {"pass": n == want and n > 0 and max(gaps, default=0) <= 1e-5,
                     "conversations": n, "rows_dtype": "float32", "max_abs_diff": max(gaps, default=None)}
# (c)
if a.recapture:
    other = load(a.recapture)
    diff = [cid for cid, (e, rows, lg) in other.items()
            if cid not in convs or not np.array_equal(rows, convs[cid][1]) or not np.array_equal(lg, convs[cid][2])
            or {k: v for k, v in e.items() if k not in ("row_offset", "messages")} !=
               {k: v for k, v in convs[cid][0].items() if k not in ("row_offset", "messages")}]
    report["c_recapture"] = {"pass": not diff and len(other) > 0, "conversations": len(other), "different": diff}
else:
    report["c_recapture"] = {"pass": None, "note": "not run"}
# (d)
tot = sum(p["labelled_messages"] for p in parts); cap = sum(p["labelled_captured"] for p in parts)
missing = [m for p in parts for m in p["missing"]]
report["d_coverage"] = {"pass": tot > 0 and cap / tot >= 0.99, "labelled": tot, "captured": cap,
                        "share": cap / tot if tot else None, "missing": missing}
# (e)
files = [p for p in out.iterdir() if p.is_file()]
report["e_storage"] = {"bytes": sum(p.stat().st_size for p in files), "files": len(files),
                       "rows": int(sum(e["rows"] for e, _, _ in convs.values())),
                       "free_bytes_on_volume": shutil.disk_usage(out).free,
                       "cleanup": f"rm -rf {out} after training (A6 e)"}
report["shards"] = [s for p in parts for s in p["shards"]]
report["runtime"] = parts[0]["runtime"] if parts else None
checks = [report["a_finite"]["pass"], report["b_trunk"]["pass"], report["d_coverage"]["pass"]]
if a.recapture:
    checks.append(report["c_recapture"]["pass"])
report["passed"] = all(checks)
(out / "CAPTURE.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: (v if k not in ("shards", "runtime") else "...") for k, v in report.items()}, indent=1)[:3000])
sys.exit(0 if report["passed"] else 1)
