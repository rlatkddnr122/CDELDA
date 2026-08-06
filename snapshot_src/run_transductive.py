"""STANDARD TRANSDUCTIVE (warm-start) 5-fold CV — the conventional LDA-literature protocol,
as a COMPLEMENT to this repo's cold-start (node hold-out) experiments.

Difference from cold-start: here ALL lncRNA and disease nodes stay in training; we randomly hold out
1/5 of the KNOWN positive PAIRS (edges) per fold, zero them in M_train, train on the partially
observed matrix (matrix completion), then score the held-out positives against the true-zero unknowns.
Two negative schemes (both reported by the literature):
  - "all" : negatives = every entry that is 0 in the FULL matrix (training positives excluded).
  - "1to1": balanced random 1:1 negative sample, averaged over 10 draws.

Models: Popularity2 (collaborative freq ceiling), MF (free embeddings, no content = logistic matrix
factorization), TwoTower (ortho content, both sides).
Content embeddings are cached -> no re-encoding. Env: CCDIFF_DATA_DIR, CCDIFF_LNC_FILE,
CCDIFF_DIS_FILE, CCDIFF_TAG, TT_*. Saves results/twoside_transductive_<TAG>.json.

HONESTY NOTE (state in reports): warm-start AUPR/AUC are inflated by node-level association-frequency
leakage and are NOT a substitute for the inductive cold-start eval; present as a complementary
upper-reference, not the headline.
"""
import os, sys, json, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from twoside_models import Popularity2, TwoTowerContent
from ccdiff_models import get_device
from ccdiff_common import SEED

DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
RES = os.path.join(os.path.dirname(__file__), "..", "results")
LNC_FILE = os.environ.get("CCDIFF_LNC_FILE", "lnc_ortho.npy")
DIS_FILE = os.environ.get("CCDIFF_DIS_FILE", "disease_emb.npy")
TAG = os.environ.get("CCDIFF_TAG", "canon")
K = 5
SCHEMES = ["all", "1to1"]
LOG = open(os.path.join(RES, f"run_log_transductive_{TAG}.txt"), "w")


def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def eval_all(S, M_full, test_pos):
    """Held-out positives vs ALL true-zero unknowns (training positives are M_full==1 -> excluded)."""
    nr, nc = np.where(M_full == 0)
    pos_s = S[test_pos[:, 0], test_pos[:, 1]]
    neg_s = S[nr, nc]
    y = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    s = np.concatenate([pos_s, neg_s])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def eval_sampled(S, M_full, test_pos, ratio=1, n_draws=10, seed=SEED):
    rng = np.random.default_rng(seed)
    zr, zc = np.where(M_full == 0)
    n_pos = len(test_pos)
    pos_s = S[test_pos[:, 0], test_pos[:, 1]]
    n_neg = min(ratio * n_pos, len(zr))
    aucs, auprs = [], []
    for _ in range(n_draws):
        sel = rng.choice(len(zr), size=n_neg, replace=False)
        neg_s = S[zr[sel], zc[sel]]
        y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        s = np.concatenate([pos_s, neg_s])
        aucs.append(roc_auc_score(y, s)); auprs.append(average_precision_score(y, s))
    return float(np.mean(aucs)), float(np.mean(auprs))


def ctors(dev):
    return {
        "Popularity": lambda: Popularity2(),
        "MF (free emb)": lambda: TwoTowerContent(content_l=False, content_d=False, device=dev),
        "TwoTower (content)": lambda: TwoTowerContent(content_l=True, content_d=True, device=dev),
    }


def main():
    dev = get_device()
    M = np.load(os.path.join(DATA, "M.npy")).astype(np.float32)
    Clnc = np.load(os.path.join(DATA, LNC_FILE)).astype(np.float32)
    Cdis = np.load(os.path.join(DATA, DIS_FILE)).astype(np.float32)
    n_l, n_d = M.shape
    all_l, all_d = np.arange(n_l), np.arange(n_d)
    pos = np.argwhere(M > 0)
    folds = np.array_split(np.random.default_rng(SEED).permutation(len(pos)), K)
    names = list(ctors(dev).keys())
    log(f"device={dev} tag={TAG} | M{M.shape} pos={len(pos)} density={100*M.mean():.2f}% "
        f"content={LNC_FILE} | TRANSDUCTIVE (warm-start) 5-fold")

    acc = {n: {s: {"auc": [], "aupr": []} for s in SCHEMES} for n in names}
    for fi, fold in enumerate(folds):
        test_pos = pos[fold]
        M_train = M.copy()
        M_train[test_pos[:, 0], test_pos[:, 1]] = 0.0     # hide held-out positives (nodes stay warm)
        log(f"\n#### fold {fi+1}/{K}  held-out positives={len(test_pos)} ####")
        for name, ctor in ctors(dev).items():
            t0 = time.time()
            mdl = ctor().fit(M_train, Clnc, Cdis, all_l, all_d)   # ALL nodes seen, masked matrix
            S = mdl.predict()
            a_all, ap_all = eval_all(S, M, test_pos)
            a_s, ap_s = eval_sampled(S, M, test_pos)
            acc[name]["all"]["auc"].append(a_all); acc[name]["all"]["aupr"].append(ap_all)
            acc[name]["1to1"]["auc"].append(a_s); acc[name]["1to1"]["aupr"].append(ap_s)
            log(f"  {name:<20} ALL AUC={a_all:.4f} AUPR={ap_all:.4f} | "
                f"1:1 AUC={a_s:.4f} AUPR={ap_s:.4f} ({time.time()-t0:.1f}s)")

    summary = {}
    for n in names:
        summary[n] = {}
        for s in SCHEMES:
            summary[n][s] = {
                "AUC": {"mean": float(np.mean(acc[n][s]["auc"])), "std": float(np.std(acc[n][s]["auc"]))},
                "AUPR": {"mean": float(np.mean(acc[n][s]["aupr"])), "std": float(np.std(acc[n][s]["aupr"]))},
            }
    out = {"dataset": TAG,
           "protocol": "TRANSDUCTIVE warm-start: 5-fold random positive-pair hold-out; all nodes in "
                       "training; negatives=true zeros of full matrix; 'all' & '1:1' schemes",
           "content": {"lncRNA": LNC_FILE, "disease": DIS_FILE}, "seed": SEED, "device": str(dev),
           "n_pos": int(len(pos)), "summary": summary}
    json.dump(out, open(os.path.join(RES, f"twoside_transductive_{TAG}.json"), "w"), indent=2)
    log("\n=== summary mean over 5 folds ===")
    log(f"  {'model':<20}{'ALL_AUC':>9}{'ALL_AUPR':>10}{'1:1_AUC':>9}{'1:1_AUPR':>10}")
    for n in names:
        a, o = summary[n]["all"], summary[n]["1to1"]
        log(f"  {n:<20}{a['AUC']['mean']:>9.4f}{a['AUPR']['mean']:>10.4f}"
            f"{o['AUC']['mean']:>9.4f}{o['AUPR']['mean']:>10.4f}")
    log(f"\nSaved -> results/twoside_transductive_{TAG}.json")


if __name__ == "__main__":
    main()
