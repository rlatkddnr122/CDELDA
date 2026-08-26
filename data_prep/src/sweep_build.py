"""Build the 16 RNADisease k-core threshold-sweep datasets (lnc and dis cut-offs in {2,3,4,5}).

For each (min_lnc, min_dis) variant, writes ccdiff/data_rd_l{L}d{D}/ with:
  M.npy, lnc_names.txt, disease_doids.txt, disease_texts.json, lncrna_seq.json (subset),
  lnc_ortho.npy (RE-INDEXED from data_rd/lnc_ortho.npy by name; zeros for nodes w/o content),
  prep_rd_report.json
Disease content (disease_emb.npy) is produced separately by embed_disease.py per variant.

Re-indexing lnc_ortho is EXACT: RNA-FM/struct/expr are per-node functions of the (identical)
sequence/gene, so a lncRNA's content row is the same in every variant; new nodes w/o sequence
get a zero row (identical to how the 799 seq-missing nodes are already handled in data_rd).
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json
import numpy as np

from data_prep.src.prep_rd import parse_obo, norm, kcore  # reuse canonical logic
import pandas as pd

ROOT = _ROOT
XL = os.path.join(ROOT, "data_rd_raw", "RNADiseasev4.0_RNA-disease_experiment_lncRNA.xlsx")
OBO = os.path.join(ROOT, "data_raw", "doid.obo")
SRC = os.path.join(ROOT, "data_rd")  # source of content rows (l2/d10, fully built)
# 16-variant sweep (lnc/dis k-core 2..5); a variant whose M.npy already exists is
# skipped inside the loop, so pre-built variants are never clobbered.
COMBOS = [(ml, md) for ml in (2, 3, 4, 5) for md in (2, 3, 4, 5)]


def main():
    df = pd.read_excel(XL, dtype=str).fillna("")
    d = df[df["specise"] == "Homo sapiens"].copy()
    d["lnc"] = d["RNA Symbol"].map(norm); d["doid"] = d["DO ID"].map(norm)
    d = d[(d["lnc"] != "") & (d["doid"] != "")]
    terms, alt = parse_obo(OBO)

    def resolve(x):
        return terms.get(x) or terms.get(alt.get(x, ""))
    d = d[d["doid"].map(lambda x: resolve(x) is not None)]
    pairs = d[["lnc", "doid"]].drop_duplicates()

    # source content + its node order
    src_lnc = open(os.path.join(SRC, "lnc_names.txt")).read().splitlines()
    src_ortho = np.load(os.path.join(SRC, "lnc_ortho.npy")).astype(np.float32)
    src_orthoS = np.load(os.path.join(SRC, "lnc_orthoS.npy")).astype(np.float32) \
        if os.path.exists(os.path.join(SRC, "lnc_orthoS.npy")) else None
    row_of = {n: i for i, n in enumerate(src_lnc)}
    seq_all = json.load(open(os.path.join(SRC, "lncrna_seq.json")))
    Odim = src_ortho.shape[1]

    for ml, md in COMBOS:
        out = os.path.join(ROOT, f"data_rd_l{ml}d{md}")
        if os.path.exists(os.path.join(out, "M.npy")):
            print(f"[l{ml}d{md}] M.npy exists, skip (avoid clobber) -> {out}")
            continue
        os.makedirs(out, exist_ok=True)
        core = kcore(pairs, ml, md)
        lncs = sorted(core["lnc"].unique()); doids = sorted(core["doid"].unique())
        li = {n: i for i, n in enumerate(lncs)}; di = {n: i for i, n in enumerate(doids)}
        M = np.zeros((len(lncs), len(doids)), np.float32)
        for l, s in core.itertuples(index=False):
            M[li[l], di[s]] = 1.0
        assert int(M.sum()) == len(core)

        # re-index lnc_ortho by name (zeros for nodes not in source content)
        ortho = np.zeros((len(lncs), Odim), np.float32)
        n_hit = 0
        for n, i in li.items():
            j = row_of.get(n)
            if j is not None:
                ortho[i] = src_ortho[j]; n_hit += 1
        orthoS = None
        if src_orthoS is not None:
            orthoS = np.zeros((len(lncs), src_orthoS.shape[1]), np.float32)
            for n, i in li.items():
                j = row_of.get(n)
                if j is not None:
                    orthoS[i] = src_orthoS[j]

        # disease texts (rich DO name+def+syn), like prep_rd
        texts = {}
        for doid in doids:
            t = resolve(doid); parts = [t["name"]]
            if t["def"]:
                parts.append(t["def"])
            if t["syn"]:
                parts.append("Synonyms: " + "; ".join(t["syn"][:6]))
            texts[doid] = ". ".join(p for p in parts if p)
        seq_sub = {n: seq_all[n] for n in lncs if n in seq_all}

        np.save(os.path.join(out, "M.npy"), M)
        np.save(os.path.join(out, "lnc_ortho.npy"), ortho)
        if orthoS is not None:
            np.save(os.path.join(out, "lnc_orthoS.npy"), orthoS)
        open(os.path.join(out, "lnc_names.txt"), "w").write("\n".join(lncs))
        open(os.path.join(out, "disease_doids.txt"), "w").write("\n".join(doids))
        json.dump(texts, open(os.path.join(out, "disease_texts.json"), "w"),
                  ensure_ascii=False, indent=0)
        json.dump(seq_sub, open(os.path.join(out, "lncrna_seq.json"), "w"))
        dis_freq = M.sum(0); lnc_freq = M.sum(1)
        report = {
            "dataset": f"RNADisease v4.0 core, k-core(lnc>={ml}, dis>={md})",
            "filter": {"kcore_min_lnc_diseases": ml, "kcore_min_dis_lncRNAs": md},
            "core": {"n_lnc": int(M.shape[0]), "n_dis": int(M.shape[1]),
                     "n_assoc": int(M.sum()), "density": float(M.mean()),
                     "lnc_ortho_content_rows": int(n_hit),
                     "lnc_ortho_zero_rows": int(len(lncs) - n_hit),
                     "dis_freq": {"min": int(dis_freq.min()), "max": int(dis_freq.max()),
                                  "mean": float(dis_freq.mean())},
                     "lnc_freq": {"min": int(lnc_freq.min()), "max": int(lnc_freq.max()),
                                  "mean": float(lnc_freq.mean())}},
            "content": {"lnc_ortho": "re-indexed by name from data_rd/lnc_ortho.npy (702d RNA-FM+struct+expr)",
                        "disease_emb": "produced by embed_disease.py (S-BioBERT 768d) on disease_texts.json"},
        }
        json.dump(report, open(os.path.join(out, "prep_rd_report.json"), "w"), indent=2)
        print(f"[l{ml}d{md}] {M.shape[0]}x{M.shape[1]} | {int(M.sum())} assoc | "
              f"ortho content rows {n_hit}/{len(lncs)} (zero {len(lncs)-n_hit}) | "
              f"dis new-text {sum(1 for x in doids if x not in set(open(os.path.join(SRC,'disease_doids.txt')).read().splitlines()))} -> {out}")


if __name__ == "__main__":
    main()
