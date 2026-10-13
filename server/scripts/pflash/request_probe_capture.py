#!/usr/bin/env python3
"""Request-probe FEATURE CAPTURE driver (SPEC A6): block-14 rows from the pinned runtime.

Runs the drafter-only pflash_daemon in the SERVED config (Q4_0 KV default, segment probe + u314 head, strict
selection) with the request probe on, one conversation per FRESH session (PFLASH_DRAFTER_SESSIONS=0) over its full
causal prefix, and keeps the block-14 rows of user-role and tool tokens in F32 (exactly the rows the served trunk
reads, rather than bf16) plus the runtime's own request logits (f32, the A6 probe). The runtime writes each call's rows with PFLASH_TRACE_REQUEST_FEATURES
(record "RPF2": i32 resume, n_new, hidden, f32 rows, f32 logits).

Input JSONL, one conversation per line (the A6 kit format works as is):
  {"conversation_id" | "id": str, "ids": [drafter token ids, the full rendered conversation],
   "messages": [{"index": int, "role": "user"|"tool"|"assistant"|"system"|..., "span": [b, e)}, ...]}
  role "user" = human (the request probe's role rule); role "tool" = tool results / observations.
  A message counts as labelled when it has "request_spans" or "labelled": true, else every user/tool message does.

Output (OUT dir, possibly on another host via --remote host): shard-NNNNN.rows.f32 (float32 [rows, hidden]),
shard-NNNNN.logits.f32 ([rows]), shard-NNNNN.index.jsonl (per conversation: id, n_tokens, row_offset, rows,
messages [{index, role, span, row_offset}], tokens = merged captured token spans in row order), part-*.json summary.
Rows of a conversation are its user/tool tokens in ascending token order.

  request_probe_capture.py capture --daemon BIN --drafter GGUF --scoring-head GGUF --segment-probe GGUF --request-probe GGUF
             --input convs.jsonl --out DIR [--part i/n] [--remote HOST] [--shard-mb 1024]
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

# Runtime artifacts, set from the command line in main().
DAEMON = DRAFTER = HEAD = SEGPROBE = REQPROBE = None
CAPTURE_ROLES = ("user", "tool")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def merged(spans):
    out = []
    for b, e in sorted(spans):
        if out and b <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        elif e > b:
            out.append([b, e])
    return out


class Daemon:
    def __init__(self, workdir, env_extra):
        self.dump = Path(workdir) / "features.rpf"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PFLASH_", "LUCE_KV"))}
        env.update({
            "PFLASH_SCORING_HEAD_GGUF": str(HEAD), "PFLASH_SEGMENT_PROBE_GGUF": str(SEGPROBE),
            "PFLASH_SELECT_MODE": "budget_only", "PFLASH_SELECT_CHUNK_SIZE": "256", "PFLASH_SELECT_SCORE": "density",
            "PFLASH_SELECT_SEGMENTS": "probe", "PFLASH_SELECT_QUERY_PARSER": "latest_user",
            "PFLASH_SELECT_QUERY_TOKENS": "128", "PFLASH_SELECT_PARAGRAPH_JOIN": "0",
            "PFLASH_SELECT_QUERY_PROBE": "1", "PFLASH_REQUEST_PROBE_GGUF": str(REQPROBE),
            "PFLASH_DRAFTER_SESSIONS": "0", "PFLASH_DAEMON_KEEP_X1000": "1000",
            "PFLASH_TRACE_REQUEST_FEATURES": str(self.dump),
        })
        env.update(env_extra)
        self.env = env
        rfd, wfd = os.pipe()
        self.log = open(Path(workdir) / "daemon.log", "a")
        self.proc = subprocess.Popen([str(DAEMON), str(DRAFTER), f"--stream-fd={wfd}"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env,
                                     pass_fds=(wfd,), text=True)
        os.close(wfd)
        self.stream = os.fdopen(rfd, "rb")
        if "ready" not in self._line(["ready"]):
            raise RuntimeError("pflash_daemon did not start (see daemon.log)")

    def _line(self, words):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return ""
            if line.startswith("[pflash-daemon]") and any(w in line for w in words):
                return line.strip()

    def capture(self, ids, human, path):
        """Rows [S, hidden] f32 and request logits [S] of the whole prompt, from a fresh session."""
        self.dump.unlink(missing_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<i4").tobytes())
        cmd = f"request 1 {len(ids)} 0 0 {len(human)}" + "".join(f" {b} {e}" for b, e in human) + f" {path}\n"
        self.proc.stdin.write(cmd)
        self.proc.stdin.flush()
        status = self._line([" done", " failed"])
        while struct.unpack("<i", self.stream.read(4))[0] != -1:
            pass
        if "done" not in status:
            raise RuntimeError(status or "daemon exited")
        buf = self.dump.read_bytes()
        self.dump.unlink()
        if buf[:4] != b"RPF2":
            raise RuntimeError("bad feature record")
        resume, n, hidden = struct.unpack_from("<3i", buf, 4)
        if resume != 0 or n != len(ids) or 16 + 4 * n * (hidden + 1) != len(buf):
            raise RuntimeError(f"record covers [{resume},{resume + n}) of {len(ids)} (not one fresh full prefix)")
        rows = np.frombuffer(buf, np.float32, n * hidden, 16).reshape(n, hidden)
        logits = np.frombuffer(buf, np.float32, n, 16 + 4 * n * hidden)
        return rows, logits

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()


def add_capture_arguments(ap):
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--remote", default="")
    ap.add_argument("--shard-mb", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workdir", default="")
    ap.add_argument("--daemon", required=True, help="pflash_daemon binary")
    ap.add_argument("--drafter", required=True, help="Qwen3.5-0.8B drafter GGUF (with a .sha256 file next to it)")
    ap.add_argument("--scoring-head", required=True, help="PFLASH_SCORING_HEAD_GGUF")
    ap.add_argument("--segment-probe", required=True, help="PFLASH_SEGMENT_PROBE_GGUF")
    ap.add_argument("--request-probe", required=True, help="PFLASH_REQUEST_PROBE_GGUF")


def capture(a):
    global DAEMON, DRAFTER, HEAD, SEGPROBE, REQPROBE
    DAEMON, DRAFTER = Path(a.daemon), Path(a.drafter)
    HEAD, SEGPROBE, REQPROBE = Path(a.scoring_head), Path(a.segment_probe), Path(a.request_probe)
    pi, pn = map(int, a.part.split("/"))
    out = Path(a.out)
    work = Path(a.workdir or tempfile.mkdtemp(prefix="rpcap-"))
    work.mkdir(parents=True, exist_ok=True)
    stage = work / "stage"
    stage.mkdir(exist_ok=True)
    if a.remote:
        subprocess.run(["ssh", "-o", "ConnectTimeout=15", a.remote, f"mkdir -p {out}"], check=True)
    else:
        out.mkdir(parents=True, exist_ok=True)

    def ship(names):
        if a.remote:
            subprocess.run(["rsync", "-a", "--remove-source-files", *[str(stage / n) for n in names],
                            f"{a.remote}:{out}/"], check=True)
        else:
            for n in names:
                shutil.move(str(stage / n), str(out / n))

    convs = []
    with open(a.input) as f:
        for k, line in enumerate(f):
            if line.strip() and k % pn == pi:
                convs.append(json.loads(line))
    if a.limit:
        convs = convs[:a.limit]
    daemon = Daemon(work, {})
    shard_id, shard_rows, shards, missing, done = 0, 0, [], [], 0
    rows_f, logit_f, index_f = None, None, None
    labelled_total = labelled_captured = 0
    started = time.time()
    hidden = None

    def open_shard():
        nonlocal rows_f, logit_f, index_f, shard_rows
        base = f"part{pi}of{pn}-shard-{shard_id:05d}"
        rows_f = open(stage / f"{base}.rows.f32", "wb")
        logit_f = open(stage / f"{base}.logits.f32", "wb")
        index_f = open(stage / f"{base}.index.jsonl", "w")
        shard_rows = 0
        return base

    def close_shard(base):
        nonlocal shard_id
        for f in (rows_f, logit_f, index_f):
            f.close()
        names = [f"{base}.rows.f32", f"{base}.logits.f32", f"{base}.index.jsonl"]
        shards.append({"shard": base, "rows": shard_rows,
                       "sha256": {n: sha256(stage / n) for n in names},
                       "bytes": sum((stage / n).stat().st_size for n in names)})
        ship(names)
        shard_id += 1

    base = open_shard()
    for c in convs:
        cid = str(c.get("conversation_id", c.get("id")))
        ids = c["ids"]
        msgs = c["messages"]
        labelled = [m for m in msgs if m["role"] in CAPTURE_ROLES and
                    (("request_spans" in m) or m.get("labelled") or
                     not any(("request_spans" in x) or x.get("labelled") for x in msgs))]
        labelled_total += len(labelled)
        human = [m["span"] for m in msgs if m["role"] == "user"][::-1][:2]
        bad = [m for m in msgs if not (0 <= m["span"][0] <= m["span"][1] <= len(ids))]
        try:
            if bad:
                raise RuntimeError(f"{len(bad)} message spans outside the ids")
            rows, logits = daemon.capture(ids, human, work / "ids.bin")
        except Exception as error:
            missing.extend({"conversation_id": cid, "message": m.get("index"), "reason": str(error)[:200]}
                           for m in labelled)
            if daemon.proc.poll() is not None:
                daemon = Daemon(work, {})
            continue
        hidden = rows.shape[1]
        if not np.isfinite(rows).all() or not np.isfinite(logits).all():
            missing.extend({"conversation_id": cid, "message": m.get("index"), "reason": "non-finite rows"}
                           for m in labelled)
            continue
        cap = merged([m["span"] for m in msgs if m["role"] in CAPTURE_ROLES])
        pos = np.concatenate([np.arange(b, e) for b, e in cap]) if cap else np.zeros(0, np.int64)
        row_of = {}
        for i, p in enumerate(pos.tolist()):
            row_of[p] = i
        entry = {"conversation_id": cid, "n_tokens": len(ids), "hidden": hidden, "row_offset": shard_rows,
                 "rows": int(pos.size), "tokens": cap, "messages": []}
        for m in msgs:
            if m["role"] not in CAPTURE_ROLES:
                continue
            b, e = m["span"]
            entry["messages"].append({"index": m.get("index"), "role": m["role"], "span": [b, e],
                                      "row_offset": shard_rows + (row_of[b] if e > b else 0)})
        labelled_captured += len(labelled)
        rows_f.write(np.ascontiguousarray(rows[pos], dtype="<f4").tobytes())
        logit_f.write(np.ascontiguousarray(logits[pos], dtype="<f4").tobytes())
        index_f.write(json.dumps(entry) + "\n")
        shard_rows += int(pos.size)
        done += 1
        if shard_rows * hidden * 4 >= a.shard_mb << 20:
            close_shard(base)
            base = open_shard()
    close_shard(base)
    daemon.close()
    summary = {"rows_dtype": "float32", "part": a.part, "conversations": len(convs), "captured": done, "labelled_messages": labelled_total,
               "labelled_captured": labelled_captured,
               "coverage": labelled_captured / labelled_total if labelled_total else 1.0,
               "missing": missing, "shards": shards, "hidden": hidden, "seconds": round(time.time() - started, 1),
               "host": os.uname().nodename,
               "runtime": {"pflash_daemon_sha256": sha256(DAEMON),
                           "drafter_gguf_sha256": open(str(DRAFTER) + ".sha256").read().split()[0],
                           "scoring_head_sha256": sha256(HEAD), "segment_probe_sha256": sha256(SEGPROBE),
                           "request_probe_sha256": sha256(REQPROBE),
                           "env": {k: v for k, v in daemon.env.items() if k.startswith(("PFLASH_", "LUCE_"))}}}
    (stage / f"part{pi}of{pn}.json").write_text(json.dumps(summary, indent=1))
    ship([f"part{pi}of{pn}.json"])
    shutil.rmtree(work, ignore_errors=True) if not a.workdir else None
    print(json.dumps({k: summary[k] for k in ("part", "conversations", "captured", "coverage", "seconds")}))


def add_validation_arguments(ap):
    ap.add_argument("out")
    ap.add_argument("--recapture", default="")
    ap.add_argument("--probe", required=True, help="the request-probe GGUF the capture ran with")
    ap.add_argument("--trunk-convs", type=int, default=20)


def validate(a):
    import math
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deps/llama.cpp/gguf-py"))
    import gguf
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
    return 0 if report["passed"] else 1


def main():
    ap = argparse.ArgumentParser(description="Capture and validate request-probe feature datasets.")
    commands = ap.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture", help="Capture fresh-session feature rows and logits")
    add_capture_arguments(capture_parser)
    validate_parser = commands.add_parser("validate", help="Check a captured dataset and write CAPTURE.json")
    add_validation_arguments(validate_parser)
    args = ap.parse_args()
    if args.command == "capture":
        capture(args)
        return 0
    return validate(args)


if __name__ == "__main__":
    sys.exit(main())
