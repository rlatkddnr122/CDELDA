"""lncRNA CONTENT feature = normalized k-mer spectrum of the RNAcentral sequence.
Length-agnostic, fully intrinsic (depends only on the sequence) -> defined for cold lncRNAs.
k=3 (64) + k=4 (256) concatenated, each L2-normalized -> 320-d. Missing sequences -> zero vector
(flagged honestly; count reported). This is the standard sequence content used in LDA papers
(e.g. HGC-GAN K=3, VGAELDA k-mer); RNA-FM is a richer learned alternative (future work).

Outputs: data/lnc_kmer.npy (240 x 320), data/lnc_kmer_meta.json
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json, itertools
import numpy as np

ROOT = _ROOT
DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(ROOT, "data"))
BASES = "ACGU"


def kmer_vec(seq, k):
    idx = {"".join(t): i for i, t in enumerate(itertools.product(BASES, repeat=k))}
    v = np.zeros(len(idx), dtype=np.float32)
    s = seq.upper().replace("T", "U")
    for i in range(len(s) - k + 1):
        j = idx.get(s[i:i + k])
        if j is not None:
            v[j] += 1.0
    n = v.sum()
    if n > 0:
        v /= n                                  # frequency
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v            # L2 normalize


def main():
    names = open(os.path.join(DATA, "lnc_names.txt")).read().splitlines()
    seqs = json.load(open(os.path.join(DATA, "lncrna_seq.json")))
    feats, n_ok = [], 0
    for nm in names:
        s = seqs.get(nm, {}).get("seq")
        if s:
            v = np.concatenate([kmer_vec(s, 3), kmer_vec(s, 4)])
            n_ok += 1
        else:
            v = np.zeros(64 + 256, dtype=np.float32)
        feats.append(v)
    X = np.stack(feats).astype(np.float32)
    print(f"lnc k-mer features: {X.shape}  | sequences used={n_ok}/{len(names)} "
          f"({100*n_ok/len(names):.1f}%)  | zero-vector (missing)={len(names)-n_ok}")
    print(f"  per-dim std mean={X.std(0).mean():.4f}  nonzero rows={int((np.abs(X).sum(1)>0).sum())}")
    np.save(os.path.join(DATA, "lnc_kmer.npy"), X)
    json.dump({"dim": int(X.shape[1]), "k": [3, 4], "n_with_seq": n_ok,
               "n_total": len(names), "coverage": round(n_ok / len(names), 4)},
              open(os.path.join(DATA, "lnc_kmer_meta.json"), "w"), indent=2)
    print(f"Saved -> data/lnc_kmer.npy")


if __name__ == "__main__":
    main()
