"""Generic wandb nested-CV HP sweep for any content hero (NCF / Bilinear / Contrastive).

Same leakage guardrail as wandb_dattn_sweep: HP selected on an INNER-VALIDATION both-cold
split of the TRAINING nodes only; outer test folds never consulted. Winning config is later
re-run through the full protocol for reporting.

Run: python -m bench.wandb_hero_sweep --hero ncf --dataset l2d5 --count 60
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sklearn.metrics import average_precision_score                     # noqa: E402
from ccdiff_models import get_device                                    # noqa: E402
from bench.wandb_dattn_sweep import load, inner_split, val_c4_aupr      # noqa: E402

# hero -> (import path, class, {config_key: env_var}, param space)
HEROES = {
    "ncf": ("bench.hero_ncf", "NCFHero",
            {"m": "NCF_M", "lr": "NCF_LR", "wd": "NCF_WD", "dropout": "NCF_DROPOUT", "epochs": "NCF_EPOCHS"},
            {"m": {"values": [64, 128, 256]},
             "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 5e-3},
             "wd": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
             "dropout": {"values": [0.0, 0.1, 0.2, 0.3, 0.5]},
             "epochs": {"values": [300, 600, 1200]}}),
    "bilinear": ("bench.hero_bilinear", "BilinearHero",
            {"m": "BIL_M", "rank": "BIL_RANK", "lr": "BIL_LR", "wd": "BIL_WD",
             "dropout": "BIL_DROPOUT", "epochs": "BIL_EPOCHS"},
            {"m": {"values": [64, 128, 256]},
             "rank": {"values": [0, 8, 16, 32]},
             "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 5e-3},
             "wd": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
             "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
             "epochs": {"values": [300, 600, 1200]}}),
    "contrastive": ("bench.hero_contrastive", "ContrastiveHero",
            {"m": "CH_M", "temp": "CH_TEMP", "lr": "CH_LR", "wd": "CH_WD",
             "dropout": "CH_DROPOUT", "epochs": "CH_EPOCHS"},
            {"m": {"values": [64, 128, 256]},
             "temp": {"distribution": "log_uniform_values", "min": 0.02, "max": 0.5},
             "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 5e-3},
             "wd": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
             "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
             "epochs": {"values": [300, 600, 1200]}}),
}


def make_train(hero, M, Cl, Cd, dev, split):
    import importlib, wandb
    modpath, clsname, envmap, _ = HEROES[hero]
    Cls = getattr(importlib.import_module(modpath), clsname)
    itr_l, itr_d, vl, vd = split

    def train():
        run = wandb.init(); c = dict(wandb.config)
        for key, env in envmap.items():
            os.environ[env] = str(c[key])
        mdl = Cls(device=dev).fit(M, Cl, Cd, itr_l, itr_d)
        score = val_c4_aupr(mdl.predict(), M, vl, vd)
        wandb.log({"inner_val_c4_aupr": score})
        print(f"[run] c4={score:.4f} " + " ".join(f"{k}={c[k]}" for k in envmap), flush=True)
        run.finish()
    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", required=True, choices=list(HEROES))
    ap.add_argument("--dataset", default="l2d5")
    ap.add_argument("--count", type=int, default=60)
    a = ap.parse_args()
    import wandb
    M, Cl, Cd = load(a.dataset); dev = get_device(); split = inner_split(M)
    _, _, _, space = HEROES[a.hero]
    cfg = {"method": "bayes", "metric": {"name": "inner_val_c4_aupr", "goal": "maximize"},
           "parameters": space}
    print(f"[wandb] hero={a.hero} dataset={a.dataset} M={M.shape} dev={dev} "
          f"inner-val lnc/dis={len(split[2])}/{len(split[3])} count={a.count}", flush=True)
    sid = wandb.sweep(cfg, project=f"ccdiff-{a.hero}-hp")
    wandb.agent(sid, function=make_train(a.hero, M, Cl, Cd, dev, split), count=a.count)
    print(f"[wandb] hero={a.hero} sweep {sid} done.", flush=True)


if __name__ == "__main__":
    main()
