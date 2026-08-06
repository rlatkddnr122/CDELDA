"""VGAELDA -- COLD-EQUIPPED content variant (inductive full-node extension).

Sibling of `bench/baselines_content/vgaelda_content.py`. That -content variant
equips VGAELDA with literal intrinsic content (lncRNA expression cosine + disease
DO-DAG semantic similarity) blended with the train-subblock GIP, BUT it restricts
every similarity to the train x train block -- so held-out (cold) nodes still fall
back to the zero-feature encoding and score at the chance floor (0.500) at
both-cold (C4). That is a limitation of the FEATURE PLUMBING, not necessarily of
the VGAE architecture.

This variant runs the FAIR test: does the VGAE *architecture* itself exploit
content at both-cold if we actually hand cold nodes their content? The ONLY
change vs -content is the node-feature / adjacency construction:

  * Node features span ALL nodes' content-similarity rows to the TRAIN columns:
        lncRNA features  Xl = blend(content, GIP)[:, train_lnc]   -> (n_l, nTl)
        disease features Xd = blend(content, GIP)[:, train_dis]   -> (n_d, nTd)
    Train rows carry the SAME train-block features the -content variant trains on
    (content[train,train] blended with GIP); COLD rows carry their content
    similarity to the train nodes (GIP contributes 0 -- cold nodes have no
    observed associations). Input dim stays nTl / nTd, so the SAME trained
    encoder weights apply to any node set (inductive GCN extension).
  * The encoder is TRAINED exactly as native/-content on the TRAIN subgraph
    (features/adjacency = the train x train block), so the contract holds: labels
    and the VGAE/LP/EM objective read ONLY M[np.ix_(train_lnc, train_dis)].
  * At PREDICT the trained encoder is re-run over the FULL node graph (train +
    cold) so cold nodes receive real embeddings from their content rows; scores
    come from the same inner-product decoder, with the train block F-blended as
    native. The GIP stays train-subblock-only; content spans all rows to train
    columns.

BLEND (unchanged form):
    SL = alpha*SL_con + (1-alpha)*SL_gip ,  SD = alpha*SD_con + (1-alpha)*SD_gip
    alpha = float(os.environ.get("VGAELDA_ALPHA", "0.5"))
  SL_con/SD_con: M-independent literal content (CCDIFF_LNC_EXPR expression cosine,
  CCDIFF_DIS_SEMSIM DO-DAG semsim; else content_cosine(Clnc)/(Cdis)).
  SL_gip/SD_gip: gip_kernel of the TRAIN association sub-block only (as native).

Sub-block invariance is PRESERVED: content is M-independent, GIP is
train-subblock-only, predict uses mu (no sampling) under a fixed seed -> scrambling
any entry outside the train sub-block leaves every feature / adjacency / parameter
/ score identical (max|Delta| ~ 0). Cold nodes are now fed content, so their
scores are differentiated (nonzero, non-constant) -- the whole point of the test.

Module-level names the runner imports: NAME, build(device) -> model.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from bench.interface import SEED, subblock, gip_kernel, content_cosine   # noqa: F401

try:  # mirror TwoTowerContent's device pattern
    from ccdiff_models import get_device
except Exception:  # pragma: no cover - defensive fallback
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

NAME = "VGAELDA-contentfull (cold-equipped)"

# --- defaults (all env-overridable so the tiny-data smoke stays fast) --------
_EPOCHS = int(os.environ.get("VGAE_EPOCHS", os.environ.get("TT_EPOCHS", "150")))
_INNER = int(os.environ.get("VGAE_INNER", "5"))
_LR = float(os.environ.get("VGAE_LR", "1e-2"))
_WD = float(os.environ.get("VGAE_WD", "1e-5"))
_KL_W = float(os.environ.get("VGAE_KL", "1e-2"))
_REC_W = float(os.environ.get("VGAE_REC", "0.3"))
_LP_W = float(os.environ.get("VGAE_LP", "0.3"))
_ALPHA = float(os.environ.get("VGAE_ALPHA", "0.5"))
_LP_MIX = float(os.environ.get("VGAE_LPMIX", "0.5"))
_BETA = float(os.environ.get("VGAE_BETA", "0.5"))
_KNN = int(os.environ.get("VGAE_KNN", "15"))
_LAT_CAP = int(os.environ.get("VGAE_LAT", "32"))
_HID_CAP = int(os.environ.get("VGAE_HID", "64"))
_BLEND_ALPHA = float(os.environ.get("VGAELDA_ALPHA", "0.5"))
_EPS = 1e-8


def _load_content_npy(env_key, n, square):
    """Load an M-INDEPENDENT content/similarity matrix from an env-pointed .npy.

    square=False -> a valid file is a (n, k) expression matrix; square=True ->
    a valid (n, n) semantic-similarity matrix. Returns validated float32 or None
    (caller falls back to intrinsic content).
    """
    path = os.environ.get(env_key)
    if not path or not os.path.isfile(path):
        return None
    try:
        arr = np.asarray(np.load(path), np.float32)
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    if square:
        if arr.shape != (n, n):
            return None
    else:
        if arr.shape[0] != n or arr.shape[1] < 1:
            return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _norm_adj(K, knn):
    """Normalised GCN adjacency from a similarity kernel (kNN + self-loop + sym-norm)."""
    A = np.asarray(K, np.float64).copy()
    n = A.shape[0]
    np.fill_diagonal(A, 0.0)
    if 0 < knn < n - 1:
        idx = np.argpartition(-A, knn - 1, axis=1)[:, :knn]
        mask = np.zeros_like(A, dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        A = np.where(mask, A, 0.0)
    A = np.maximum(A, A.T)
    A = A + np.eye(n, dtype=np.float64)
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, _EPS))
    A = dinv[:, None] * A * dinv[None, :]
    return A.astype(np.float32)


def _resolvent(A_norm, alpha):
    """Closed-form label-propagation smoother P = (I - alpha * A_norm)^{-1} (R2)."""
    A = np.asarray(A_norm, np.float64)
    n = A.shape[0]
    I = np.eye(n, dtype=np.float64)
    try:
        P = np.linalg.solve(I - alpha * A, I)
    except np.linalg.LinAlgError:                     # pragma: no cover - defensive
        P = I.copy()
        term = I.copy()
        for _ in range(128):
            term = alpha * (term @ A)
            P = P + term
            if np.abs(term).max() < 1e-7:
                break
    return P.astype(np.float32)


class _GCN(nn.Module):
    """Dense GCN layer: out = A_norm @ (X @ W) + b (bias added post-aggregation)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, A, X):
        return A @ self.weight(X) + self.bias


class _VGAEEncoder(nn.Module):
    """2-layer dense GCN variational encoder -> (mu, logvar). logvar clamped."""

    def __init__(self, in_dim, hid, lat):
        super().__init__()
        self.gc1 = _GCN(in_dim, hid)
        self.gc_mu = _GCN(hid, lat)
        self.gc_lv = _GCN(hid, lat)

    def forward(self, A, X):
        h = torch.relu(self.gc1(A, X))
        mu = self.gc_mu(A, h)
        lv = torch.clamp(self.gc_lv(A, h), -10.0, 10.0)
        return mu, lv


def _kl(mu, lv):
    """Mean-over-nodes KL(N(mu, exp(lv)) || N(0, I))."""
    return -0.5 * torch.mean(torch.sum(1.0 + lv - mu.pow(2) - torch.exp(lv), dim=1))


def _reparam(mu, lv):
    return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)


class _VGAELDA:
    """Cold-equipped VGAELDA: train on the train subgraph, extend inductively to all nodes."""

    def __init__(self, device="cpu"):
        self.device = device if device is not None else get_device()
        self.epochs = _EPOCHS
        self.inner = max(1, _INNER)
        self.lr = _LR
        self.wd = _WD
        self.kl_w = _KL_W
        self.rec_w = _REC_W
        self.lp_w = _LP_W
        self.alpha = _ALPHA
        self.lp_mix = _LP_MIX
        self.beta = _BETA
        self.knn = _KNN
        self.blend_alpha = _BLEND_ALPHA
        self.seed = SEED

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        torch.manual_seed(self.seed)
        dev = self.device
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc, dtype=np.int64)
        td = np.asarray(train_dis, dtype=np.int64)
        self._tl, self._td = tl, td

        Msub = subblock(M, tl, td).astype(np.float32)     # (nTl, nTd) ONLY supervision
        nTl, nTd = Msub.shape
        k = int(min(_LAT_CAP, nTl, nTd))                  # adaptive latent dim

        # Degenerate train block -> honest all-zero floor.
        if nTl == 0 or nTd == 0 or k < 1:
            self._trained = False
            self.S = np.zeros((self.n_l, self.n_d), np.float32)
            return self

        a = self.blend_alpha

        # --- M-independent literal content (FULL, all nodes) --------------------
        expr = _load_content_npy("CCDIFF_LNC_EXPR", self.n_l, square=False)
        SL_con_full = np.asarray(content_cosine(expr) if expr is not None
                                 else content_cosine(Clnc), np.float32)   # (n_l, n_l)
        ss = _load_content_npy("CCDIFF_DIS_SEMSIM", self.n_d, square=True)
        SD_con_full = np.asarray(ss if ss is not None
                                 else content_cosine(Cdis), np.float32)   # (n_d, n_d)

        # --- GIP (train association sub-block ONLY; identical to native) --------
        SL_gip = gip_kernel(Msub)                         # (nTl, nTl)
        SD_gip = gip_kernel(Msub.T)                       # (nTd, nTd)

        # --- TRAIN-block blended similarity (what the encoder is TRAINED on) ----
        SL_tr = (a * SL_con_full[np.ix_(tl, tl)] + (1.0 - a) * SL_gip).astype(np.float32)
        SD_tr = (a * SD_con_full[np.ix_(td, td)] + (1.0 - a) * SD_gip).astype(np.float32)
        Al_np = _norm_adj(SL_tr, self.knn)
        Ad_np = _norm_adj(SD_tr, self.knn)

        # --- R2: closed-form label-propagation resolvents (train block only) ----
        Pl = torch.tensor(_resolvent(Al_np, self.alpha), device=dev)   # (nTl, nTl)
        Pd = torch.tensor(_resolvent(Ad_np, self.alpha), device=dev)   # (nTd, nTd)

        # Train-subgraph encoder inputs: features = blended rows, adjacency = graph.
        Fl = torch.tensor(SL_tr, device=dev)              # (nTl, nTl) node features
        Fd = torch.tensor(SD_tr, device=dev)              # (nTd, nTd) node features
        Al = torch.tensor(Al_np, device=dev)
        Ad = torch.tensor(Ad_np, device=dev)
        SLt = torch.tensor(SL_tr, device=dev)             # VGAE self-recon targets
        SDt = torch.tensor(SD_tr, device=dev)
        Yblk = torch.tensor(Msub, device=dev)             # (nTl, nTd) association targets

        # --- FULL-node inductive inputs (COLD-EQUIPPED; used only at predict) ---
        # Features: all nodes' content rows to TRAIN columns; train rows also carry
        # the GIP blend (cold rows have no GIP -> content only). Train rows equal
        # the training features SL_tr, so the trained weights extend consistently.
        Xl_np = (a * SL_con_full[:, tl]).astype(np.float32)           # (n_l, nTl)
        Xl_np[tl] += (1.0 - a) * SL_gip
        Xd_np = (a * SD_con_full[:, td]).astype(np.float32)           # (n_d, nTd)
        Xd_np[td] += (1.0 - a) * SD_gip
        # Full graph: content similarity everywhere; train x train block blended
        # with GIP (equals the train subgraph on that block).
        SL_full = (a * SL_con_full).astype(np.float32).copy()         # (n_l, n_l)
        SL_full[np.ix_(tl, tl)] += (1.0 - a) * SL_gip
        SD_full = (a * SD_con_full).astype(np.float32).copy()         # (n_d, n_d)
        SD_full[np.ix_(td, td)] += (1.0 - a) * SD_gip
        Al_full = torch.tensor(_norm_adj(SL_full, self.knn), device=dev)   # (n_l, n_l)
        Ad_full = torch.tensor(_norm_adj(SD_full, self.knn), device=dev)   # (n_d, n_d)
        Xl_full = torch.tensor(Xl_np, device=dev)                    # (n_l, nTl)
        Xd_full = torch.tensor(Xd_np, device=dev)                    # (n_d, nTd)

        hid_l = int(min(_HID_CAP, max(k, nTl)))
        hid_d = int(min(_HID_CAP, max(k, nTd)))
        self.enc_l = _VGAEEncoder(nTl, hid_l, k).to(dev)
        self.enc_d = _VGAEEncoder(nTd, hid_d, k).to(dev)

        params = list(self.enc_l.parameters()) + list(self.enc_d.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

        pos = float(Msub.sum())
        neg = float(Msub.size - pos)
        pos_w = torch.tensor([neg / (pos + 1.0)], device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        mse = nn.MSELoss()

        n_outer = max(1, self.epochs // self.inner)
        self.enc_l.train(); self.enc_d.train()
        loss_val = 0.0
        for _ in range(n_outer):
            # -- E-step (LP branch): F from a seed folding in the VGAE's scores. ---
            with torch.no_grad():
                mu_l, _ = self.enc_l(Al, Fl)
                mu_d, _ = self.enc_d(Ad, Fd)
                vg_pred = torch.sigmoid(mu_l @ mu_d.T)
                seed = Yblk + self.lp_mix * vg_pred * (1.0 - Yblk)
                F = Pl @ seed @ Pd
                F = F / (F.max() + _EPS)

            # -- M-step (VGAE branch): inner Adam steps, regularised by F. ---------
            for _ in range(self.inner):
                opt.zero_grad()
                mu_l, lv_l = self.enc_l(Al, Fl)
                mu_d, lv_d = self.enc_d(Ad, Fd)
                z_l = _reparam(mu_l, lv_l)
                z_d = _reparam(mu_d, lv_d)
                logits = z_l @ z_d.T
                loss = bce(logits, Yblk)
                loss = loss + self.rec_w * (mse(torch.sigmoid(z_l @ z_l.T), SLt)
                                            + mse(torch.sigmoid(z_d @ z_d.T), SDt))
                loss = loss + self.kl_w * (_kl(mu_l, lv_l) + _kl(mu_d, lv_d))
                loss = loss + self.lp_w * mse(torch.sigmoid(logits), F)
                loss.backward()
                opt.step()
            loss_val = float(loss.item())
        self.final_loss = loss_val

        # --- inductive full-node scoring + F-blend on the train block -----------
        self.enc_l.eval(); self.enc_d.eval()
        with torch.no_grad():
            # Train-time embeddings for the converged propagated labels F.
            mu_l, _ = self.enc_l(Al, Fl)                  # (nTl, k)
            mu_d, _ = self.enc_d(Ad, Fd)                  # (nTd, k)
            vg_pred = torch.sigmoid(mu_l @ mu_d.T)
            seed = Yblk + self.lp_mix * vg_pred * (1.0 - Yblk)
            F_final = Pl @ seed @ Pd
            F_final = F_final / (F_final.max() + _EPS)    # (nTl, nTd) in [0, 1]

            # COLD-EQUIPPED: re-run the SAME trained encoder over ALL nodes so cold
            # nodes get real content-derived embeddings (inductive extension).
            mu_l_full, _ = self.enc_l(Al_full, Xl_full)   # (n_l, k)
            mu_d_full, _ = self.enc_d(Ad_full, Xd_full)   # (n_d, k)

            S = torch.sigmoid(mu_l_full @ mu_d_full.T)    # (n_l, n_d) latent decoder
            # Blend the co-trained latent score with propagated labels on the warm
            # train sub-block (cold rows/cols keep the pure content-latent score).
            ti = torch.tensor(tl, device=dev, dtype=torch.long)
            tj = torch.tensor(td, device=dev, dtype=torch.long)
            blk = (1.0 - self.beta) * S[ti][:, tj] + self.beta * F_final
            S[ti.unsqueeze(1), tj.unsqueeze(0)] = blk

            assert torch.isfinite(S).all(), "VGAELDA-contentfull produced non-finite scores"
            self.S = S.detach().cpu().numpy().astype(np.float32)
        self._trained = True
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _VGAELDA(device)
