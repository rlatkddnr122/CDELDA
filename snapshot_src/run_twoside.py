"""Two-sided (lncRNA seq k-mer  +  disease text) cold-start across THREE inductive scenarios:
  disease-cold, lncRNA-cold, both-cold (true bilateral C4).

Models: Popularity2 (content-free), TwoTower (both content), TwoTower-lncOnly, TwoTower-disOnly.
5 paired folds, MPS. Eval over the held-out block only. Outputs: results/twoside.json, run_log_twoside.txt
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from twoside_common import load, folds, scenario_indices, eval_block
from twoside_models import Popularity2, TwoTowerContent
from ccdiff_models import get_device

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TAG = os.environ.get("CCDIFF_TAG", "kmer")              # 'kmer' or 'rnafm'
LNC_FILE = os.environ.get("CCDIFF_LNC_FILE", "lnc_kmer.npy")
LOG = open(os.path.join(RES, f"run_log_twoside_{TAG}.txt"), "w")
SCN = [("disease", "disease"), ("lncRNA", "lncRNA"), ("both", "disease")]  # (scenario, query_axis)
METRICS = ["AUC_1to1", "AUPR_1to1", "AUPR_all", "Recall@20", "Recall@50", "personalization"]


def log(*a):
    m = " ".join(str(x) for x in a); print(m); LOG.write(m + "\n"); LOG.flush()


def make_models(dev):
    return {
        "Popularity": lambda: Popularity2(),
        "TwoTower (both)": lambda: TwoTowerContent(content_l=True, content_d=True, device=dev),
        "TwoTower lnc-only": lambda: TwoTowerContent(content_l=True, content_d=False, device=dev),
        "TwoTower dis-only": lambda: TwoTowerContent(content_l=False, content_d=True, device=dev),
    }


def main():
    dev = get_device(); log(f"device={dev} torch={torch.__version__}")
    M, Clnc, Cdis = load()
    log(f"M{M.shape} #assoc={int(M.sum())} | lnc k-mer{Clnc.shape} | dis emb{Cdis.shape}")
    lfolds = folds(M.shape[0], 5); dfolds = folds(M.shape[1], 5)
    names = list(make_models(dev).keys())
    out = {}

    for scn, qaxis in SCN:
        log(f"\n################ scenario = {scn}-cold (query={qaxis}) ################")
        acc = {n: {m: [] for m in METRICS} for n in names}
        npos_log = []
        for fi in range(5):
            tr_l, tr_d, ev_l, ev_d = scenario_indices(scn, lfolds[fi], dfolds[fi],
                                                      n_lnc=M.shape[0], n_dis=M.shape[1])
            models = make_models(dev)
            for name, ctor in models.items():
                t0 = time.time()
                mdl = ctor().fit(M, Clnc, Cdis, tr_l, tr_d)
                S = mdl.predict()
                assert not np.isnan(S).any(), f"{name} NaN"
                r = eval_block(S, M, ev_l, ev_d, qaxis)
                for m in METRICS:
                    acc[name][m].append(r[m])
                if name == "Popularity":
                    npos_log.append(r["n_pos"])
            log(f"  [fold {fi+1}] block: n_query={r['n_query']} n_cand={r['n_cand']} "
                f"n_pos={r['n_pos']}  ({time.time()-t0:.1f}s/model last)")

        out[scn] = {n: {m: {"mean": float(np.nanmean(acc[n][m])), "std": float(np.nanstd(acc[n][m]))}
                        for m in METRICS} for n in names}
        log(f"\n  --- {scn}-cold summary (mean over 5 folds; eval-block positives "
            f"mean={np.mean(npos_log):.0f}) ---")
        log(f"  {'model':<20}" + "".join(f"{m:>12}" for m in METRICS))
        for n in names:
            s = out[scn][n]
            log(f"  {n:<20}" + "".join(f"{s[m]['mean']:>12.4f}" for m in METRICS))

    # label derived from the ACTUAL feature file loaded (no hardcoded fabrication)
    LNC_LABELS = {"lnc_rnafm.npy": "RNA-FM encoder (640d)", "lnc_kmer.npy": "3+4-mer (320d)",
                  "lnc_multiview.npy": "k-mer ⊕ RNA-FM multiview (960d)",
                  "lnc_ortho.npy": "RNA-FM ⊕ structure ⊕ expression (orthogonal 702d)",
                  "lnc_orthoS.npy": "RNA-FM ⊕ structure (high-coverage 648d, no expr)"}
    lnc_desc = LNC_LABELS.get(LNC_FILE, f"{Clnc.shape[1]}-d")
    DIS_FILE = os.environ.get("CCDIFF_DIS_FILE", "disease_emb.npy")
    from twoside_common import DATA as _DATADIR
    _dm = os.path.join(_DATADIR, "disease_emb_meta.json")
    try:
        dis_src = json.load(open(_dm)).get("source", "")
    except Exception:
        dis_src = ""
    dis_desc = (f"{dis_src} ({Cdis.shape[1]}d) [{DIS_FILE}]" if dis_src
                else f"disease content ({Cdis.shape[1]}d) [{DIS_FILE}]")
    ds_label = (f"LDAformer canonical 240x412/2697" if M.shape == (240, 412)
                else f"HGCMLDA 861x253/4517" if M.shape == (861, 253)
                else f"M{M.shape[0]}x{M.shape[1]}/{int(M.sum())}")
    result = {"dataset": ds_label,
              "content": {"lncRNA": f"{lnc_desc} [{LNC_FILE}]",
                          "disease": dis_desc},
              "device": dev, "seed": 2026, "tag": TAG, "scenarios": out}
    json.dump(result, open(os.path.join(RES, f"twoside_{TAG}.json"), "w"), indent=2)
    log(f"\nSaved -> results/twoside_{TAG}.json, results/run_log_twoside_{TAG}.txt")


if __name__ == "__main__":
    main()
