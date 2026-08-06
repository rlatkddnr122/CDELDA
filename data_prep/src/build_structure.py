"""lncRNA secondary-STRUCTURE content view (orthogonal to k-mer/RNA-FM sequence-composition views).
ViennaRNA MFE fold on the sequence (capped to CAP nt for tractability on long lncRNAs) -> compact
structure feature vector. Intrinsic (from sequence) -> defined for cold lncRNAs.

Features per lncRNA: [mfe/len, paired_frac, gc_frac, n_stem_transitions/len, mean_unpaired_run/len,
                      mean_paired_run/len, frac_'('-vs-')'(balance), len_used/CAP]  -> 8-d
Missing sequence -> zeros. Saves CCDIFF_DATA_DIR/lnc_struct.npy.
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json
import numpy as np
import RNA

DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(_ROOT, "data"))
CAP = 1500


def runs(mask):
    """mean run length of True and False in a boolean list."""
    if not mask:
        return 0.0, 0.0
    t, f, cur, val = [], [], 1, mask[0]
    for m in mask[1:]:
        if m == val:
            cur += 1
        else:
            (t if val else f).append(cur); cur, val = 1, m
    (t if val else f).append(cur)
    return (np.mean(t) if t else 0.0), (np.mean(f) if f else 0.0)


def feats(seq):
    s = seq.upper().replace("T", "U")[:CAP]
    s = "".join(c if c in "ACGU" else "A" for c in s)  # ViennaRNA needs valid; rare N->A
    if len(s) < 10:
        return np.zeros(8, np.float32)
    struct, mfe = RNA.fold(s)
    L = len(s)
    paired = [c != "." for c in struct]
    pf = sum(paired) / L
    gc = (s.count("G") + s.count("C")) / L
    trans = sum(1 for i in range(1, L) if paired[i] != paired[i - 1]) / L  # stem/loop transitions
    pr, ur = runs(paired)                                # mean paired-run, unpaired-run
    bal = abs(struct.count("(") - struct.count(")")) / L
    return np.array([mfe / L, pf, gc, trans, pr / L, ur / L, bal, L / CAP], np.float32)


def _feat_or_zero(s):
    return feats(s) if s else np.zeros(8, np.float32)


def main():
    names = open(os.path.join(DATA, "lnc_names.txt")).read().splitlines()
    seqs = json.load(open(os.path.join(DATA, "lncrna_seq.json")))
    seqlist = [seqs.get(nm, {}).get("seq") for nm in names]
    n = len(names)
    n_ok = int(sum(1 for s in seqlist if s))
    nproc = int(os.environ.get("CCDIFF_NPROC", "1"))                 # parallel ViennaRNA folding
    CHUNK = int(os.environ.get("CCDIFF_STRUCT_CHUNK", "2000"))        # checkpoint granularity

    # ---- crash-resilient checkpoint: resume partial folding after a host crash/reboot ----
    part_f = os.path.join(DATA, "lnc_struct_partial.npy")
    done_f = os.path.join(DATA, "lnc_struct_done.json")
    X = np.zeros((n, 8), np.float32)
    start = 0
    if os.path.exists(part_f) and os.path.exists(done_f):
        try:
            prev = np.load(part_f)
            d = json.load(open(done_f))
            if prev.shape == (n, 8) and 0 < d.get("done", 0) <= n:
                X = prev.astype(np.float32); start = int(d["done"])
                print(f"[resume] checkpoint found -> skip first {start}/{n} (이어서 폴딩)", flush=True)
        except Exception as e:
            print(f"[resume] checkpoint unreadable ({e}); start from 0", flush=True)

    if start >= n:
        print(f"[resume] all {n} already folded; finalizing", flush=True)
    else:
        import time as _t
        t0 = _t.time()
        pool = None
        if nproc > 1:
            import multiprocessing as mp
            pool = mp.Pool(nproc)
        try:
            for b in range(start, n, CHUNK):
                e = min(b + CHUNK, n)
                seg = seqlist[b:e]
                res = pool.map(_feat_or_zero, seg, chunksize=16) if pool else [_feat_or_zero(s) for s in seg]
                X[b:e] = np.stack(res).astype(np.float32)
                np.save(part_f, X)                                   # atomic-ish checkpoint
                json.dump({"done": e, "n": n}, open(done_f, "w"))
                print(f"  folded {e}/{n} ({100*e/n:.1f}%, {_t.time()-t0:.0f}s) [checkpoint saved]", flush=True)
        finally:
            if pool is not None:
                pool.close(); pool.join()

    print(f"structure feats: {X.shape} | folded {n_ok}/{n} (cap {CAP}nt) | "
          f"per-dim std {np.round(X.std(0), 3)}")
    np.save(os.path.join(DATA, "lnc_struct.npy"), X)
    for f in (part_f, done_f):                                       # clean checkpoint on success
        if os.path.exists(f):
            os.remove(f)
    json.dump({"dim": 8, "cap": CAP, "n_folded": n_ok, "n": len(names),
               "feats": ["mfe/len", "paired_frac", "gc", "transitions/len", "paired_run/len",
                         "unpaired_run/len", "balance", "len/cap"]},
              open(os.path.join(DATA, "lnc_struct_meta.json"), "w"), indent=2)
    print(f"Saved -> {DATA}/lnc_struct.npy")


if __name__ == "__main__":
    main()
