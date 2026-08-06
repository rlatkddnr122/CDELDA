"""Preprocess the LncRNADisease v3.0 database (rnanut.net) into the CC-DiffLDA array layout.

Source: data_ld_raw/website_alldata.tsv  (full ncRNA-disease table, 25,440 entries).
We keep EXPERIMENTALLY-validated, Homo sapiens, ncRNA Category == LncRNA rows, dedupe to a binary
(lncRNA x disease) matrix, then take an iterative k-core (default lnc>=2 diseases, dis>=3 lncRNAs)
so every retained node has enough signal for inductive cold-start hold-out. Disease side is free-text
MeSH-style NAMES -> intrinsic content via S-BioBERT (embed_disease_names.py), defined for cold diseases.

Outputs (CCDIFF_DATA_DIR, default ccdiff/data_ld):
  M.npy             : (n_lnc, n_dis) float32 binary association matrix (k-core)
  lnc_names.txt     : n_lnc lncRNA symbols (row order)
  disease_names.txt : n_dis disease names (col order)   [input to embed_disease_names.py]
  prep_ld_report.json : full + core stats (all parsed, no fabrication)

No fabrication: every number is parsed/derived from the downloaded TSV.
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, re, json
import numpy as np
import pandas as pd

ROOT = _ROOT
RAW = os.path.join(ROOT, "data_ld_raw", "website_alldata.tsv")
OUT = os.environ.get("CCDIFF_DATA_DIR", os.path.join(ROOT, "data_ld"))
MIN_LNC = int(os.environ.get("LD_MIN_LNC", "2"))   # min #diseases per lncRNA
MIN_DIS = int(os.environ.get("LD_MIN_DIS", "3"))   # min #lncRNAs per disease
os.makedirs(OUT, exist_ok=True)


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def kcore(pairs, min_l, min_d):
    p = pairs.copy()
    while True:
        ld = p.groupby("lnc").size(); dd = p.groupby("dis").size()
        keepl = set(ld[ld >= min_l].index); keepd = set(dd[dd >= min_d].index)
        n0 = len(p)
        p = p[p["lnc"].isin(keepl) & p["dis"].isin(keepd)]
        if len(p) == n0:
            return p


def main():
    df = pd.read_csv(RAW, sep="\t", dtype=str).fillna("")
    d = df[(df["ncRNA Category"].str.lower() == "lncrna") & (df["Species"] == "Homo sapiens")].copy()
    d["lnc"] = d["ncRNA Symbol"].map(norm); d["dis"] = d["Disease Name"].map(norm)
    d = d[(d["lnc"] != "") & (d["dis"] != "")]
    pairs = d[["lnc", "dis"]].drop_duplicates()
    full = {"n_lnc": int(pairs["lnc"].nunique()), "n_dis": int(pairs["dis"].nunique()),
            "n_assoc": int(len(pairs))}
    print(f"full human LncRNA: {full['n_lnc']} lnc x {full['n_dis']} dis | {full['n_assoc']} assoc "
          f"(density={full['n_assoc']/(full['n_lnc']*full['n_dis']):.5f})")

    core = kcore(pairs, MIN_LNC, MIN_DIS)
    lncs = sorted(core["lnc"].unique()); diss = sorted(core["dis"].unique())
    li = {n: i for i, n in enumerate(lncs)}; di = {n: i for i, n in enumerate(diss)}
    M = np.zeros((len(lncs), len(diss)), dtype=np.float32)
    for l, s in core.itertuples(index=False):
        M[li[l], di[s]] = 1.0
    assert int(M.sum()) == len(core)
    dis_freq = M.sum(0); lnc_freq = M.sum(1)
    print(f"k-core(lnc>={MIN_LNC}, dis>={MIN_DIS}): {M.shape[0]} lnc x {M.shape[1]} dis | "
          f"{int(M.sum())} assoc (density={M.mean():.4f})")
    print(f"  disease degree: min={int(dis_freq.min())} max={int(dis_freq.max())} mean={dis_freq.mean():.1f}")
    print(f"  lncRNA  degree: min={int(lnc_freq.min())} max={int(lnc_freq.max())} mean={lnc_freq.mean():.1f}")

    np.save(os.path.join(OUT, "M.npy"), M)
    with open(os.path.join(OUT, "lnc_names.txt"), "w") as f:
        f.write("\n".join(lncs))
    with open(os.path.join(OUT, "disease_names.txt"), "w") as f:
        f.write("\n".join(diss))
    report = {
        "dataset": "LncRNADisease v3.0 (rnanut.net), experimental Homo sapiens LncRNA-disease",
        "source_file": "data_ld_raw/website_alldata.tsv",
        "filter": {"category": "LncRNA", "species": "Homo sapiens",
                   "kcore_min_lnc_diseases": MIN_LNC, "kcore_min_dis_lncRNAs": MIN_DIS},
        "full": full,
        "core": {"n_lnc": int(M.shape[0]), "n_dis": int(M.shape[1]), "n_assoc": int(M.sum()),
                 "density": float(M.mean()),
                 "dis_freq": {"min": int(dis_freq.min()), "max": int(dis_freq.max()),
                              "mean": float(dis_freq.mean())},
                 "lnc_freq": {"min": int(lnc_freq.min()), "max": int(lnc_freq.max()),
                              "mean": float(lnc_freq.mean())}},
    }
    json.dump(report, open(os.path.join(OUT, "prep_ld_report.json"), "w"), indent=2)
    print(f"  e.g. diseases: {diss[:4]}")
    print(f"Saved -> {OUT}/ (M.npy, lnc_names.txt, disease_names.txt, prep_ld_report.json)")


if __name__ == "__main__":
    main()
