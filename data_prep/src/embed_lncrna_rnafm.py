"""lncRNA CONTENT via the RNA-FM sequence ENCODER (pretrained foundation model), replacing k-mer.
RNA-FM context <=1024; long lncRNAs (max ~205k nt) are split into non-overlapping 1022-nt windows,
each mean-pooled over tokens, then mean-pooled across windows -> one 640-d embedding per lncRNA.
Per-seq window cap bounds compute (and the one 205k-nt outlier); truncation reported honestly.

GPU: MPS. Missing sequence -> zero vector (flagged). Saves data/lnc_rnafm.npy (240 x 640).
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json, time
import numpy as np
import torch

ROOT = _ROOT
DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(ROOT, "data"))
WIN = 1022
MAX_WIN = 30                      # cap windows/seq (~30 kb) to bound compute
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL = os.environ.get("CCDIFF_RNA_MODEL", "multimolecule/rnafm")   # e.g. multimolecule/rinalmo-giga
OUTFILE = os.environ.get("CCDIFF_RNA_OUT", "lnc_rnafm.npy")


def clean(seq):
    s = seq.upper().replace("T", "U")
    return "".join(c if c in "ACGUN" else "N" for c in s)


def windows(s):
    chunks = [s[i:i + WIN] for i in range(0, len(s), WIN)]
    return chunks[:MAX_WIN], len(chunks)


@torch.no_grad()
def main():
    from multimolecule import RnaTokenizer
    tok = RnaTokenizer.from_pretrained(MODEL)
    if MODEL == "multimolecule/rnafm":
        from multimolecule import RnaFmModel
        model = RnaFmModel.from_pretrained(MODEL).to(DEV).eval()
    else:
        from multimolecule import AutoModel
        model = AutoModel.from_pretrained(MODEL).to(DEV).eval()
    H = model.config.hidden_size
    names = open(os.path.join(DATA, "lnc_names.txt")).read().splitlines()
    seqs = json.load(open(os.path.join(DATA, "lncrna_seq.json")))

    feats = np.zeros((len(names), H), dtype=np.float32)
    n_ok, n_trunc, t0 = 0, 0, time.time()
    for i, nm in enumerate(names):
        s = seqs.get(nm, {}).get("seq")
        if not s:
            continue
        chunks, total = windows(clean(s))
        if total > MAX_WIN:
            n_trunc += 1
        # batch a sequence's chunks
        embs = []
        for b in range(0, len(chunks), 8):
            batch = chunks[b:b + 8]
            enc = tok(batch, return_tensors="pt", padding=True).to(DEV)
            out = model(**enc).last_hidden_state            # (B,L,H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)   # masked mean over tokens
            embs.append(pooled)
        feats[i] = torch.cat(embs, 0).mean(0).cpu().numpy()  # mean over windows
        n_ok += 1
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(names)} embedded (ok={n_ok}, {time.time()-t0:.0f}s)")

    print(f"RNA-FM lncRNA embeddings: {feats.shape} | encoded {n_ok}/{len(names)} "
          f"({100*n_ok/len(names):.1f}%) | seqs truncated(>{MAX_WIN*WIN}nt)={n_trunc} "
          f"| {time.time()-t0:.0f}s")
    print(f"  per-dim std mean={feats.std(0).mean():.4f}  nonzero rows={int((np.abs(feats).sum(1)>0).sum())}")
    np.save(os.path.join(DATA, OUTFILE), feats)
    json.dump({"encoder": MODEL, "dim": int(H), "n_encoded": n_ok,
               "n_total": len(names), "coverage": round(n_ok / len(names), 4),
               "window": WIN, "max_windows": MAX_WIN, "n_truncated": n_trunc, "device": DEV},
              open(os.path.join(DATA, OUTFILE.replace(".npy", "_meta.json")), "w"), indent=2)
    print(f"Saved -> {DATA}/{OUTFILE}")


if __name__ == "__main__":
    main()
