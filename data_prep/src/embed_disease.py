"""Embed the 412 disease DO-definition texts into fixed-length vectors = disease CONTENT feature.
This content is INTRINSIC (from the Disease Ontology, not from lncRNA associations), so it is
defined even for a held-out (cold) disease -> the conditioning signal that makes true C4 solvable.

GPU: uses MPS if available. Saves ccdiff/data/disease_emb.npy (412 x d), aligned to disease_doids.txt.
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json
import numpy as np
import torch

ROOT = _ROOT
OUT = os.environ.get("CCDIFF_DATA_DIR", os.path.join(ROOT, "data"))

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CANDIDATES = [
    "pritamdeka/S-BioBert-snli-multinli-stsb",   # biomedical sentence encoder (768-d)
    "sentence-transformers/all-MiniLM-L6-v2",    # robust general fallback (384-d)
]


def main():
    from sentence_transformers import SentenceTransformer
    doids = open(os.path.join(OUT, "disease_doids.txt")).read().splitlines()
    texts_map = json.load(open(os.path.join(OUT, "disease_texts.json")))
    texts = [texts_map[d] if texts_map.get(d) else d for d in doids]  # blanks -> use id string
    assert len(texts) == len(doids) and len(texts) > 0, f"text/doid mismatch: {len(texts)} vs {len(doids)}"

    model, used = None, None
    for name in CANDIDATES:
        try:
            print(f"loading {name} on {DEVICE} ...")
            model = SentenceTransformer(name, device=DEVICE)
            used = name
            break
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {str(e)[:160]}")
    if model is None:
        raise RuntimeError("no embedding model could be loaded")

    emb = model.encode(texts, batch_size=32, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    emb = emb.astype(np.float32)
    print(f"disease embeddings: {emb.shape} via {used} (device={DEVICE})")
    # sanity: embeddings should vary across diseases (not collapsed)
    print(f"  per-dim std mean={emb.std(0).mean():.4f}  | pairwise cos(0,1)={float(emb[0]@emb[1]):.3f}")
    np.save(os.path.join(OUT, "disease_emb.npy"), emb)
    with open(os.path.join(OUT, "disease_emb_meta.json"), "w") as f:
        json.dump({"model": used, "dim": int(emb.shape[1]), "n": int(emb.shape[0]), "device": DEVICE,
                   "normalized": True}, f, indent=2)
    print(f"Saved -> {OUT}/disease_emb.npy  (model recorded in disease_emb_meta.json)")


if __name__ == "__main__":
    main()
