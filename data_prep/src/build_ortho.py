"""Build ORTHOGONAL multiview content by concatenating genuinely different modalities (each L2-norm):
  lncRNA  = RNA-FM(seq) ⊕ structure ⊕ expression          -> lnc_ortho.npy
  disease = S-BioBERT(text) ⊕ disease-gene                 -> disease_ortho.npy
HPO excluded (only ~6% coverage on this cancer-heavy set). Each view L2-normalized so none dominates.
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json
import numpy as np

DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(_ROOT, "data"))


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def load(f):
    return np.load(os.path.join(DATA, f)).astype(np.float32)


def main():
    # lncRNA: sequence (RNA-FM) + structure + expression
    rnafm = load("lnc_rnafm.npy"); struct = load("lnc_struct.npy"); expr = load("lnc_expr.npy")
    lnc = np.concatenate([l2(rnafm), l2(struct), l2(expr)], 1).astype(np.float32)
    np.save(os.path.join(DATA, "lnc_ortho.npy"), lnc)
    print(f"lnc_ortho: {lnc.shape}  = RNA-FM{rnafm.shape[1]} ⊕ struct{struct.shape[1]} ⊕ expr{expr.shape[1]}")
    # high-coverage-only variant (drop expression) -> tests whether a low-coverage expr view is noise.
    lncS = np.concatenate([l2(rnafm), l2(struct)], 1).astype(np.float32)
    np.save(os.path.join(DATA, "lnc_orthoS.npy"), lncS)
    print(f"lnc_orthoS: {lncS.shape}  = RNA-FM{rnafm.shape[1]} ⊕ struct{struct.shape[1]}  (no expr)")

    # disease: text (S-BioBERT) + disease-gene  -- ONLY if a disease_gene view exists.
    # Optimal config keeps disease = S-BioBERT alone (disease-gene's low coverage = noise), so
    # when disease_gene.npy is absent (e.g. big dataset) we skip disease_ortho and use disease_emb.
    gene_path = os.path.join(DATA, "disease_gene.npy")
    dis_meta = {"dis_views": ["sbiobert_only (disease_emb.npy)"], "dis_dim": None}
    if os.path.exists(gene_path):
        text = load("disease_emb.npy"); gene = load("disease_gene.npy")
        dis = np.concatenate([l2(text), l2(gene)], 1).astype(np.float32)
        np.save(os.path.join(DATA, "disease_ortho.npy"), dis)
        print(f"disease_ortho: {dis.shape}  = S-BioBERT{text.shape[1]} ⊕ gene{gene.shape[1]}")
        dis_meta = {"dis_views": ["sbiobert", "disease_gene"], "dis_dim": int(dis.shape[1])}
    else:
        print("disease_gene.npy absent -> skip disease_ortho (use disease_emb.npy = S-BioBERT only)")

    json.dump({"lnc_views": ["rnafm", "struct", "expr"], "lnc_dim": int(lnc.shape[1]),
               **dis_meta, "hpo_excluded": "coverage ~6% on cancer-heavy set"},
              open(os.path.join(DATA, "ortho_meta.json"), "w"), indent=2)
    print(f"Saved -> {DATA}/lnc_ortho.npy" + (", disease_ortho.npy" if os.path.exists(gene_path) else ""))


if __name__ == "__main__":
    main()
