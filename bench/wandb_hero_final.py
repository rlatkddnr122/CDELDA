"""E0 — wandb nested-CV sweep to DETERMINE the final hero.

Searches both the interaction/HP AND the disease-encoder mode:
  mode = "dot"  -> frozen-content TwoTowerContent (dot product), search m/wd/dropout/lr/epochs
  mode = "lora" -> PEFTDiseaseHero (LoRA-adapted S-BioBERT disease encoder), search r/lr/wd/epochs/m

Objective = inner-validation both-cold AUPR on the dev fold's training partition ONLY
(outer test folds never consulted; the winner is re-verified on outer 5-seed before locking).

Run: python -m bench.wandb_hero_final --dataset l2d5 --count 120
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(_REPO, "data"))
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _PAPER = os.path.dirname(_HERE)
for _p in (_PAPER, os.path.join(_PAPER, "snapshot_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from ccdiff_models import get_device                                    # noqa: E402
from bench.wandb_dattn_sweep import load, inner_split, val_c4_aupr      # noqa: E402

_DATA_ROOT = _DATA


def build_config():
    return {
        "method": "bayes",
        "metric": {"name": "inner_val_c4_aupr", "goal": "maximize"},
        "parameters": {
            "mode":    {"values": ["dot", "lora"]},
            "m":       {"values": [64, 128, 256]},
            "lr":      {"distribution": "log_uniform_values", "min": 1e-4, "max": 5e-3},
            "wd":      {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
            "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},     # dot only
            "epochs":  {"values": [300, 500, 1000]},         # dot
            "peft_epochs": {"values": [60, 100, 150]},       # lora
            "lora_r":  {"values": [4, 8, 16]},               # lora
        },
    }


def make_train(dataset, M, Cl, Cd, dev, split):
    import wandb
    itr_l, itr_d, vl, vd = split
    d = os.path.join(_DATA_ROOT, f"data_rd_{dataset}")

    def train():
        run = wandb.init(); c = wandb.config
        if c.mode == "dot":
            from twoside_models import TwoTowerContent
            for k, v in {"TT_M": c.m, "TT_LR": c.lr, "TT_WD": c.wd,
                         "TT_DROPOUT": c.dropout, "TT_EPOCHS": c.epochs}.items():
                os.environ[k] = str(v)
            mdl = TwoTowerContent(content_l=True, content_d=True, device=dev)
        else:
            from bench.hero_peft import PEFTDiseaseHero
            os.environ["CCDIFF_DIS_DOIDS"] = f"{d}/disease_doids.txt"
            os.environ["CCDIFF_DIS_TEXTS"] = f"{d}/disease_texts.json"
            for k, v in {"PEFT_M": c.m, "PEFT_LR": c.lr, "PEFT_WD": c.wd,
                         "PEFT_EPOCHS": c.peft_epochs, "PEFT_R": c.lora_r,
                         "TT_EPOCHS": c.peft_epochs}.items():
                os.environ[k] = str(v)
            mdl = PEFTDiseaseHero(device=dev)
        S = mdl.fit(M, Cl, Cd, itr_l, itr_d).predict()
        score = val_c4_aupr(S, M, vl, vd)
        wandb.log({"inner_val_c4_aupr": score})
        extra = f"r={c.lora_r} pep={c.peft_epochs}" if c.mode == "lora" else f"do={c.dropout} ep={c.epochs}"
        print(f"[run] c4={score:.4f} mode={c.mode} m={c.m} lr={c.lr:.1e} wd={c.wd:.1e} {extra}", flush=True)
        run.finish()
    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="l2d5")
    ap.add_argument("--count", type=int, default=120)
    a = ap.parse_args()
    import wandb
    M, Cl, Cd = load(a.dataset); dev = get_device(); split = inner_split(M)
    print(f"[wandb] E0 hero-final dataset={a.dataset} M={M.shape} dev={dev} "
          f"inner-val lnc/dis={len(split[2])}/{len(split[3])} count={a.count}", flush=True)
    sid = wandb.sweep(build_config(), project="ccdiff-hero-final")
    wandb.agent(sid, function=make_train(a.dataset, M, Cl, Cd, dev, split), count=a.count)
    print(f"[wandb] E0 sweep {sid} done.", flush=True)


if __name__ == "__main__":
    main()
