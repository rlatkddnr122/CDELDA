"""wandb nested-CV hyperparameter sweep for the dual-attention hero (DualAttnHero).

LEAKAGE GUARDRAIL: hyperparameters are selected on an INNER-VALIDATION split of the
TRAINING nodes only. We take one dev fold's training partition, split it into
inner-train / inner-val as a fresh both-cold split, fit on inner-train and score
inner-val both-cold AUPR. The outer test folds used for the reported headline are
NEVER consulted for selection. The winning config is later re-run through the full
5-fold protocol for reporting (separate step).

Log-scale ranges for lr/wd/temp; generous epoch budget searched discretely.
Run:  WANDB_MODE=offline python -m bench.wandb_dattn_sweep --dataset l2d10 --count 80
      (drop WANDB_MODE=offline to log online via ~/.netrc creds)
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, sys, json, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sklearn.metrics import average_precision_score                     # noqa: E402
from twoside_common import folds, scenario_indices                      # noqa: E402
from ccdiff_models import get_device                                    # noqa: E402

_DATA_ROOT = _DATA
SEED = 2026


def load(ds):
    d = os.path.join(_DATA_ROOT, f"data_rd_{ds}")
    M = np.load(f"{d}/M.npy").astype(np.float32)
    Cl = np.load(f"{d}/lnc_ortho.npy").astype(np.float32)
    Cd = np.load(f"{d}/disease_emb.npy").astype(np.float32)
    return M, Cl, Cd


def inner_split(M, seed=SEED, dev_fold=0, val_frac=0.2):
    """Dev fold's TRAIN nodes -> inner-train / inner-val (both-cold). No outer-test contact."""
    n_l, n_d = M.shape
    lf = folds(n_l, 5, seed=seed); df = folds(n_d, 5, seed=seed)
    tr_l = np.setdiff1d(np.arange(n_l), lf[dev_fold])   # dev-fold training lncRNAs
    tr_d = np.setdiff1d(np.arange(n_d), df[dev_fold])
    rng = np.random.default_rng(seed + 1)
    vl = rng.choice(tr_l, size=max(1, int(len(tr_l) * val_frac)), replace=False)
    vd = rng.choice(tr_d, size=max(1, int(len(tr_d) * val_frac)), replace=False)
    itr_l = np.setdiff1d(tr_l, vl); itr_d = np.setdiff1d(tr_d, vd)
    return itr_l, itr_d, vl, vd


def val_c4_aupr(S, M, vl, vd, seed=SEED):
    sub = M[np.ix_(vl, vd)].ravel().astype(int); sc = S[np.ix_(vl, vd)].ravel()
    pos = np.where(sub == 1)[0]; neg = np.where(sub == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    neg = np.random.default_rng(seed).choice(neg, size=min(len(pos), len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    return float(average_precision_score(sub[idx], sc[idx]))


def build_config():
    return {
        "method": "bayes",
        "metric": {"name": "inner_val_c4_aupr", "goal": "maximize"},
        "parameters": {
            "h":       {"values": [64, 128, 256]},
            "heads":   {"values": [1, 2, 4, 8]},
            "temp":    {"distribution": "log_uniform_values", "min": 0.05, "max": 2.0},
            "lr":      {"distribution": "log_uniform_values", "min": 1e-4, "max": 5e-3},
            "wd":      {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
            "epochs":  {"values": [300, 600, 1200, 2000]},
        },
    }


def make_train(M, Cl, Cd, dev, split):
    itr_l, itr_d, vl, vd = split
    from bench.dattn import DualAttnHero
    import wandb

    def train():
        run = wandb.init()
        c = wandb.config
        os.environ["DATTN_H"] = str(c.h); os.environ["DATTN_HEADS"] = str(c.heads)
        os.environ["DATTN_TEMP"] = str(c.temp); os.environ["TT_LR"] = str(c.lr)
        os.environ["TT_WD"] = str(c.wd); os.environ["TT_EPOCHS"] = str(c.epochs)
        mdl = DualAttnHero(h=c.h, heads=c.heads, epochs=c.epochs, lr=c.lr, wd=c.wd,
                           temp=c.temp, device=dev).fit(M, Cl, Cd, itr_l, itr_d)
        S = mdl.predict()
        score = val_c4_aupr(S, M, vl, vd)
        wandb.log({"inner_val_c4_aupr": score})
        print(f"[run] c4={score:.4f} h={c.h} heads={c.heads} temp={c.temp:.3f} "
              f"lr={c.lr:.1e} wd={c.wd:.1e} ep={c.epochs}", flush=True)
        run.finish()
    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="l2d10")
    ap.add_argument("--count", type=int, default=80)
    ap.add_argument("--project", default="ccdiff-dattn-hp")
    a = ap.parse_args()
    import wandb
    M, Cl, Cd = load(a.dataset); dev = get_device()
    split = inner_split(M)
    print(f"[wandb] dataset={a.dataset} M={M.shape} dev={dev} "
          f"inner-train lnc/dis={len(split[0])}/{len(split[1])} "
          f"inner-val lnc/dis={len(split[2])}/{len(split[3])} count={a.count}", flush=True)
    sweep_id = wandb.sweep(build_config(), project=a.project)
    wandb.agent(sweep_id, function=make_train(M, Cl, Cd, dev, split), count=a.count)
    print(f"[wandb] sweep {sweep_id} done; inspect best config in the wandb UI / offline dir.", flush=True)


if __name__ == "__main__":
    main()
