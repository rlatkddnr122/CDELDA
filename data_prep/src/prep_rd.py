"""Preprocess RNADisease v4.0 (rnadisease.org) experimental lncRNA-disease into the CC-DiffLDA layout,
with RICH disease content (Disease Ontology name+definition+synonyms) — the same content style as the
canonical 240x412 experiment, so content gets a FAIR test at large scale (unlike the disease-NAME-only
LncRNADisease run). Large-scale, independent of the canonical/LncRNADisease sources.

Source: data_rd_raw/RNADiseasev4.0_RNA-disease_experiment_lncRNA.xlsx (76,871 rows; has a DO ID column).
Keep Homo sapiens rows whose DO ID resolves in doid.obo, dedupe to a binary (lncRNA x DOID-disease)
matrix, then iterative k-core (default lnc>=2 diseases / dis>=10 lncRNAs via env LD_MIN_LNC/LD_MIN_DIS).

Outputs (CCDIFF_DATA_DIR, default ccdiff/data_rd):
  M.npy, lnc_names.txt, disease_doids.txt, disease_texts.json, prep_rd_report.json
disease_texts.json -> embed_disease.py (S-BioBERT) -> disease_emb.npy. No fabrication.
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, re, json
import numpy as np
import pandas as pd

ROOT = _ROOT
XL = os.path.join(ROOT, "data_rd_raw", "RNADiseasev4.0_RNA-disease_experiment_lncRNA.xlsx")
OBO = os.path.join(ROOT, "data_raw", "doid.obo")
OUT = os.environ.get("CCDIFF_DATA_DIR", os.path.join(ROOT, "data_rd"))
MIN_LNC = int(os.environ.get("LD_MIN_LNC", "2"))
MIN_DIS = int(os.environ.get("LD_MIN_DIS", "10"))
os.makedirs(OUT, exist_ok=True)


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_obo(path):
    terms, alt = {}, {}
    cur = None

    def flush(c):
        if c and c.get("id"):
            terms[c["id"]] = c
    for raw in open(path, errors="ignore"):
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            flush(cur)
            cur = {"id": None, "name": "", "def": "", "syn": [], "alt": []} if line == "[Term]" else None
            continue
        if cur is None:
            continue
        if line.startswith("id: "):
            cur["id"] = line[4:].strip()
        elif line.startswith("name: "):
            cur["name"] = line[6:].strip()
        elif line.startswith("def: "):
            m = re.search(r'"(.*?)"', line[5:]); cur["def"] = m.group(1) if m else ""
        elif line.startswith("synonym: "):
            m = re.search(r'"(.*?)"', line[9:])
            if m:
                cur["syn"].append(m.group(1))
        elif line.startswith("alt_id: "):
            cur["alt"].append(line[8:].strip())
    flush(cur)
    for t in terms.values():
        for a in t["alt"]:
            alt[a] = t["id"]
    return terms, alt


def kcore(p, ml, md):
    while True:
        a = p.groupby("lnc").size(); b = p.groupby("doid").size()
        kl = set(a[a >= ml].index); kd = set(b[b >= md].index); n0 = len(p)
        p = p[p["lnc"].isin(kl) & p["doid"].isin(kd)]
        if len(p) == n0:
            return p


def main():
    df = pd.read_excel(XL, dtype=str).fillna("")
    d = df[df["specise"] == "Homo sapiens"].copy()
    d["lnc"] = d["RNA Symbol"].map(norm); d["doid"] = d["DO ID"].map(norm)
    d = d[(d["lnc"] != "") & (d["doid"] != "")]
    terms, alt = parse_obo(OBO)

    def resolve(doid):
        return terms.get(doid) or terms.get(alt.get(doid, ""))
    d = d[d["doid"].map(lambda x: resolve(x) is not None)]
    pairs = d[["lnc", "doid"]].drop_duplicates()
    full = {"n_lnc": int(pairs["lnc"].nunique()), "n_dis": int(pairs["doid"].nunique()),
            "n_assoc": int(len(pairs))}
    print(f"full human lncRNA (DOID-resolvable): {full['n_lnc']} lnc x {full['n_dis']} dis | "
          f"{full['n_assoc']} assoc (density={full['n_assoc']/(full['n_lnc']*full['n_dis']):.5f})")

    core = kcore(pairs, MIN_LNC, MIN_DIS)
    lncs = sorted(core["lnc"].unique()); doids = sorted(core["doid"].unique())
    li = {n: i for i, n in enumerate(lncs)}; di = {n: i for i, n in enumerate(doids)}
    M = np.zeros((len(lncs), len(doids)), dtype=np.float32)
    for l, s in core.itertuples(index=False):
        M[li[l], di[s]] = 1.0
    assert int(M.sum()) == len(core)
    dis_freq = M.sum(0); lnc_freq = M.sum(1)
    print(f"k-core(lnc>={MIN_LNC}, dis>={MIN_DIS}): {M.shape[0]} lnc x {M.shape[1]} dis | "
          f"{int(M.sum())} assoc (density={M.mean():.4f})")
    print(f"  disease degree: min={int(dis_freq.min())} max={int(dis_freq.max())} mean={dis_freq.mean():.1f}")
    print(f"  lncRNA  degree: min={int(lnc_freq.min())} max={int(lnc_freq.max())} mean={lnc_freq.mean():.1f}")

    # rich disease text from Disease Ontology (name. definition. Synonyms: ...) — canonical content style
    texts = {}
    for doid in doids:
        t = resolve(doid)
        parts = [t["name"]]
        if t["def"]:
            parts.append(t["def"])
        if t["syn"]:
            parts.append("Synonyms: " + "; ".join(t["syn"][:6]))
        texts[doid] = ". ".join(p for p in parts if p)
    covered = sum(1 for v in texts.values() if v)

    np.save(os.path.join(OUT, "M.npy"), M)
    with open(os.path.join(OUT, "lnc_names.txt"), "w") as f:
        f.write("\n".join(lncs))
    with open(os.path.join(OUT, "disease_doids.txt"), "w") as f:
        f.write("\n".join(doids))
    with open(os.path.join(OUT, "disease_texts.json"), "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=0)
    report = {
        "dataset": "RNADisease v4.0 (rnadisease.org), experimental Homo sapiens lncRNA-disease, DO-text",
        "source_file": "data_rd_raw/RNADiseasev4.0_RNA-disease_experiment_lncRNA.xlsx",
        "content": "Disease Ontology name+definition+synonyms (rich, canonical-style) via doid.obo",
        "filter": {"species": "Homo sapiens", "require_resolvable_DOID": True,
                   "kcore_min_lnc_diseases": MIN_LNC, "kcore_min_dis_lncRNAs": MIN_DIS},
        "full": full,
        "core": {"n_lnc": int(M.shape[0]), "n_dis": int(M.shape[1]), "n_assoc": int(M.sum()),
                 "density": float(M.mean()), "disease_text_coverage": covered,
                 "dis_freq": {"min": int(dis_freq.min()), "max": int(dis_freq.max()), "mean": float(dis_freq.mean())},
                 "lnc_freq": {"min": int(lnc_freq.min()), "max": int(lnc_freq.max()), "mean": float(lnc_freq.mean())}},
    }
    json.dump(report, open(os.path.join(OUT, "prep_rd_report.json"), "w"), indent=2)
    ex = doids[:2]
    for dq in ex:
        print(f"  e.g. {dq}: {texts[dq][:120]}...")
    print(f"disease text coverage: {covered}/{len(doids)}")
    print(f"Saved -> {OUT}/ (M.npy, lnc_names.txt, disease_doids.txt, disease_texts.json, prep_rd_report.json)")


if __name__ == "__main__":
    main()
