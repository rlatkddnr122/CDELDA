"""External-dataset validation of the C-dis case study.

The case study is scored against RNADisease v4.0, the corpus the model was
trained and evaluated on. To ask whether its rankings agree with evidence held
outside that corpus we re-score the same rankings against two independent
resources:

  LncRNADisease  manual curation of lncRNA-disease associations
                 (Bao et al. 2019; Lin et al. 2024). Diseases are matched by
                 MeSH-style name pattern, since it does not carry DO ids.
  DISEASES       the text-mining channel of the Jensen-lab DISEASES resource,
                 which mines all of PubMed and is keyed directly on DO ids, so
                 the target cancers match exactly.

Two quantities are reported per cancer and source:

  agreement      hits@20 / hits@50 against the external association set
  novel-confirmed  hits among lncRNAs the external source associates with the
                 cancer but RNADisease v4.0 does NOT, i.e. predictions that our
                 own corpus scores as negatives and an outside resource
                 supports

Neither source is independent experimental evidence: all three curate or mine
the same primary literature. Agreement means a prediction is supported by a
body of evidence assembled separately from the one used to fit and score the
model.

Outputs results_case_study/external_validation.json
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os
import sys
import json
from math import comb

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = f"{_DATA}/data_rd_l2d5"
RAW = _DATA
LD = os.path.join(RAW, "data_ld_raw", "website_alldata.tsv")
DIS = os.path.join(RAW, "data_raw", "diseases_textmining.tsv")
RES = os.path.join(_PAPER, "results_case_study")
MAIN = "TwoTower-PEFT-disease (LoRA S-BioBERT)"
SEED = 2026
DIS_MIN = 2.0        # DISEASES text-mining confidence floor

NAMES = {"DOID:684": "Hepatocellular carcinoma", "DOID:10534": "Gastric cancer",
         "DOID:1612": "Breast cancer", "DOID:9256": "Colorectal cancer",
         "DOID:3908": "Non-small cell lung cancer"}
# LncRNADisease uses MeSH-style labels, so each target is matched by pattern.
# Broad on purpose: a subtype counts as support for its parent cancer.
PATTERNS = {
    "DOID:684":   r"hepatocell|liver neoplasm|liver cancer",
    "DOID:10534": r"stomach|gastric",
    "DOID:1612":  r"breast",
    "DOID:9256":  r"colorect|colon neoplasm|rectal neoplasm",
    "DOID:3908":  r"non-small|nonsmall|non small",
}


def hg(k, N, K, n):
    if K == 0 or n == 0 or n > N:
        return 1.0
    return float(sum(comb(K, i) * comb(N - K, n - i)
                     for i in range(k, min(K, n) + 1)) / comb(N, n))


def main():
    res = json.load(open(os.path.join(RES, "case_cdis_l2d5.json")))
    names = [l.rstrip("\n") for l in open(os.path.join(DATA, "lnc_names.txt"))]
    doids = [l.strip() for l in open(os.path.join(DATA, "disease_doids.txt"))]
    M = np.load(os.path.join(DATA, "M.npy"))
    upper = {s.upper(): i for i, s in enumerate(names)}
    n_l = len(names)

    # ---- source 1: LncRNADisease -------------------------------------------
    ld = pd.read_csv(LD, sep="\t", dtype=str)
    ld = ld[(ld["Species"] == "Homo sapiens")
            & (ld["ncRNA Category"].str.lower() == "lncrna")]
    ld_sets, ld_pmid = {}, {}
    for do, pat in PATTERNS.items():
        sub = ld[ld["Disease Name"].str.contains(pat, case=False, na=False, regex=True)]
        idx = set()
        for _, r in sub.iterrows():
            i = upper.get(str(r["ncRNA Symbol"]).strip().upper())
            if i is not None:
                idx.add(i)
                ld_pmid.setdefault((do, i), set()).add(str(r["PubMed ID"]))
        ld_sets[do] = idx

    # ---- source 2: DISEASES text-mining channel ----------------------------
    dm = pd.read_csv(DIS, sep="\t", header=None, dtype=str,
                     names=["ens", "symbol", "doid", "disease", "z", "conf", "url"])
    dm["conf"] = pd.to_numeric(dm["conf"], errors="coerce")
    dm = dm[dm["conf"] >= DIS_MIN]
    dm_sets = {}
    for do in NAMES:
        sub = dm[dm["doid"] == do]
        dm_sets[do] = {upper[s] for s in sub["symbol"].str.upper()
                       if s in upper}

    out = {"sources": {
        "LncRNADisease": "manual curation, matched by disease-name pattern",
        "DISEASES (text-mining)": f"Jensen-lab text mining over PubMed, DO-id keyed, "
                                  f"confidence >= {DIS_MIN}"},
        "caveat": ("All three resources curate or mine the same primary literature, so "
                   "agreement is support from a separately assembled body of evidence, "
                   "not independent experimental confirmation."),
        "cancers": {}}

    print(f"{'cancer':<30}{'source':<26}{'|set|':>7}{'h@20':>6}{'h@50':>6}"
          f"{'novel h@20':>12}{'novel set':>11}")
    for do, disp in NAMES.items():
        j = doids.index(do)
        ours = set(np.flatnonzero(M[:, j] > 0).tolist())
        top = [e["index"] for e in res["models"][MAIN][do]["top50"]]
        rec = {"doid": do, "n_ours": len(ours), "sources": {}}
        for src, sets in (("LncRNADisease", ld_sets), ("DISEASES (text-mining)", dm_sets)):
            ext = sets[do]
            novel = ext - ours          # external says yes, our corpus does not
            h20 = sum(1 for i in top[:20] if i in ext)
            h50 = sum(1 for i in top[:50] if i in ext)
            nv20 = sum(1 for i in top[:20] if i in novel)
            nv50 = sum(1 for i in top[:50] if i in novel)
            rec["sources"][src] = {
                "n_external": len(ext), "n_novel_vs_ours": len(novel),
                "hits@20": h20, "hits@50": h50,
                "expected@20": 20 * len(ext) / n_l,
                "p@20": hg(h20, n_l, len(ext), 20),
                "novel_hits@20": nv20, "novel_hits@50": nv50,
                "novel_symbols@50": [names[i] for i in top[:50] if i in novel],
                "pmids": {names[i]: sorted(ld_pmid.get((do, i), []))[:3]
                          for i in top[:50] if i in novel} if src == "LncRNADisease" else {},
            }
            print(f"{disp[:29]:<30}{src:<26}{len(ext):>7}{h20:>6}{h50:>6}"
                  f"{nv20:>12}{len(novel):>11}")
        out["cancers"][disp] = rec

    path = os.path.join(RES, "external_validation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
