"""VGAELDA -- variational graph auto-encoders + label propagation, co-trained by
alternating variational-EM (CONTENT-BLIND, plain-torch reproduction).

Faithful-in-STRUCTURE reproduction of VGAELDA (Shi et al., BMC Bioinformatics
2021, DOI 10.1186/s12859-021-04073-z): the paper's signature system is TWO
collaborating sub-networks *per node type* -- a variational graph autoencoder
(VGAE) that reconstructs its own similarity graph, and a label-propagation (LP)
graph autoencoder whose embedding is tied to a closed-form/iterative smoothing of
the observed associations -- optimized ALTERNATELY (variational-EM) so the two
branches mutually regularize each other. `torch_geometric` is unavailable here,
so the GCN encoders are dense from-scratch torch ops (matrices are small: 30
nodes in the smoke gate, <=5035 in the real datasets).

RATIFIED POLICY -- CONTENT-BLIND (non-negotiable): Clnc / Cdis (RNA-FM /
S-BioBERT / expression) are received by fit() only to honour THE CONTRACT
signature and are NEVER read. The ONLY signal is the Gaussian Interaction Profile
(GIP) kernel of the TRAIN association sub-block; this GIP kernel *substitutes* for
the paper's integrated (functional+GIP+semantic) similarity -- the single policy
concession. Everything else is the paper's algorithm STRUCTURE:

    Msub = subblock(M, train_lnc, train_dis)   # (nTl, nTd) ONLY supervision
    SL   = gip_kernel(Msub)     -> (nTl, nTl)   # lncRNA train-side GIP similarity
    SD   = gip_kernel(Msub.T)   -> (nTd, nTd)   # disease  train-side GIP similarity

Restored algorithm components
-----------------------------
[R1] TWO variational GAEs (one per node type). Node features = GIP-similarity
     rows; graph = kNN-sparsified, symmetrized, self-looped, symmetric-normalised
     GIP graph A_norm = D^-1/2 (A+I) D^-1/2. Each VGAE: 2-layer dense-GCN encoder
     -> (mu, logvar) [logvar clamped] -> reparam Z (train) / mu (eval) -> the
     canonical VGAE objective, reconstructing its OWN similarity graph
     sigmoid(Z Zᵀ) ~= S, plus a KL regulariser. (restores review-D3)
[R2] LABEL-PROPAGATION branch as a closed-form/iterative RESOLVENT, not a single
     hop. P = (I - alpha * A_norm)^{-1} (alpha<1) smooths to convergence on each
     side; the propagated label field is F = P_l @ seed @ P_d. (restores D2)
[R3] ALTERNATING variational-EM co-training. Outer loop alternates:
       E-step (LP branch): recompute F from a SEED = observed Msub blended with
         the VGAE's current decoded scores on the unobserved entries, propagated
         by the resolvent -> the LP branch is regularised BY the VGAE.
       M-step (VGAE branch): several inner Adam steps train the two VGAEs to (a)
         reconstruct Msub (weighted BCE inner-product decoder), (b) self-recon
         their similarity graphs, (c) KL, and (d) stay CONSISTENT with F (MSE) ->
         the VGAE is regularised BY the LP branch.
     This reciprocal reinforcement is the paper's central mechanism. (restores D1)
[R4] Decoder scores from the co-trained latents + propagated labels: a PLAIN
     inner product logits = Z_l @ Z_dᵀ (no extra learned coupling -- removes the
     review-D4 capacity add), and the final train-block score is blended with the
     converged propagated labels F.

Predict FULL (n_l, n_d)
-----------------------
Full-node latent matrices: train rows carry their encoder mu on the train GIP
graph; COLD (held-out) rows carry the deterministic zero-feature encoding (empty
GIP profile on a graph-isolated self-loop node) -- an honest near-constant floor,
NOT fabricated scores. S_full = sigmoid(Z_l_full @ Z_d_fullᵀ); the warm train
sub-block is blended with the propagated labels F. float32, all finite.

Determinism / sub-block invariance: every trained tensor derives ONLY from Msub,
so scrambling M outside the train sub-block leaves Msub / SL / SD / adjacencies /
resolvents / F and all parameters identical; with torch.manual_seed(SEED) and
mu-at-eval (no sampling in predict), fit->predict is bit-reproducible.

Module-level names the runner imports: NAME, build(device) -> model.
"""
import os

import numpy as np
import torch
import torch.nn as nn

# Leakage-safe shared helpers + the global seed. subblock is the ONLY reader of
# supervision; gip_kernel builds the train-side GIP kernels. Content helpers
# (content_cosine / cosine_sim / integrated_row_sim) are deliberately NOT
# imported: this baseline is CONTENT-BLIND.
from bench.interface import SEED, subblock, gip_kernel   # noqa: F401

try:  # mirror TwoTowerContent's device pattern
    from ccdiff_models import get_device
except Exception:  # pragma: no cover - defensive fallback
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

NAME = "VGAELDA (co-trained VGAE + label-prop resolvent, content-blind)"

# --- defaults (all env-overridable so the tiny-data smoke stays fast) --------
# Epoch budget follows the shared TT_EPOCHS bound (the runner/smoke "keep it
# fast/bounded" knob) when VGAE_EPOCHS is not explicitly set, so this baseline
# stays BOUNDED on the toy (TT_EPOCHS=5) and on real data without a private env.
# The original VGAELDA trains 500 epochs (lr 0.01, wd 1e-5, hidden 256, alpha .5).
_EPOCHS = int(os.environ.get("VGAE_EPOCHS", os.environ.get("TT_EPOCHS", "150")))  # total VGAE (M-step) updates
_INNER = int(os.environ.get("VGAE_INNER", "5"))       # M-step inner updates per E-step
_LR = float(os.environ.get("VGAE_LR", "1e-2"))        # paper lr = 1e-2
_WD = float(os.environ.get("VGAE_WD", "1e-5"))        # paper weight_decay = 1e-5
_KL_W = float(os.environ.get("VGAE_KL", "1e-2"))      # KL weight (both sides)
_REC_W = float(os.environ.get("VGAE_REC", "0.3"))     # VGAE self-recon weight (R1/D3)
_LP_W = float(os.environ.get("VGAE_LP", "0.3"))       # VGAE<->LP consistency weight (R3)
_ALPHA = float(os.environ.get("VGAE_ALPHA", "0.5"))   # resolvent smoothing (R2), <1
_LP_MIX = float(os.environ.get("VGAE_LPMIX", "0.5"))  # E-step: VGAE feedback into seed
_BETA = float(os.environ.get("VGAE_BETA", "0.5"))     # predict: latent<->LP score blend
_KNN = int(os.environ.get("VGAE_KNN", "15"))          # GIP-graph kNN sparsity
_LAT_CAP = int(os.environ.get("VGAE_LAT", "32"))      # latent-dim cap
_HID_CAP = int(os.environ.get("VGAE_HID", "64"))      # hidden-dim cap
_EPS = 1e-8


def _norm_adj(K, knn):
    """Normalised GCN adjacency from a GIP kernel (kNN-sparsify + self-loop + sym-norm).

    K : (n, n) GIP kernel (float). Returns A_norm = D^-1/2 (A + I) D^-1/2 as
    float32 (spectral radius <= 1, so the R2 resolvent is well-conditioned).
    Deterministic (top-k by value; ties broken by argpartition order).
    """
    A = np.asarray(K, np.float64).copy()
    n = A.shape[0]
    np.fill_diagonal(A, 0.0)                          # drop self before kNN
    if 0 < knn < n - 1:
        idx = np.argpartition(-A, knn - 1, axis=1)[:, :knn]
        mask = np.zeros_like(A, dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        A = np.where(mask, A, 0.0)
    A = np.maximum(A, A.T)                            # symmetrise
    A = A + np.eye(n, dtype=np.float64)               # self-loops
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, _EPS))
    A = dinv[:, None] * A * dinv[None, :]
    return A.astype(np.float32)


def _resolvent(A_norm, alpha):
    """Closed-form label-propagation smoother P = (I - alpha * A_norm)^{-1} (R2).

    A_norm is symmetric-normalised (spectral radius <= 1) and alpha < 1, so
    (I - alpha A_norm) is strictly diagonally dominant in spectrum -> invertible.
    Falls back to a truncated Neumann series sum_t (alpha A_norm)^t if the direct
    solve is numerically singular. This is the multi-hop resolvent that replaces
    the old single-hop SL@Msub@SD anchor.
    """
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
    """Content-blind VGAELDA: two co-trained variational GAEs + LP resolvent."""

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
        self.seed = SEED

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        # Clnc / Cdis are DELIBERATELY IGNORED (content-blind policy).
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

        # --- content-blind signal: train-side GIP kernels (R1 features/graphs) --
        SL = gip_kernel(Msub)                             # (nTl, nTl)
        SD = gip_kernel(Msub.T)                           # (nTd, nTd)
        Al_np = _norm_adj(SL, self.knn)
        Ad_np = _norm_adj(SD, self.knn)

        # --- R2: closed-form label-propagation resolvents (computed once) -------
        Pl = torch.tensor(_resolvent(Al_np, self.alpha), device=dev)   # (nTl, nTl)
        Pd = torch.tensor(_resolvent(Ad_np, self.alpha), device=dev)   # (nTd, nTd)

        # GCN inputs: features = GIP rows, adjacency = normalised GIP graph.
        Fl = torch.tensor(SL, device=dev)                 # (nTl, nTl) node features
        Fd = torch.tensor(SD, device=dev)                 # (nTd, nTd) node features
        Al = torch.tensor(Al_np, device=dev)
        Ad = torch.tensor(Ad_np, device=dev)
        SLt = torch.tensor(SL, device=dev)                # VGAE self-recon targets
        SDt = torch.tensor(SD, device=dev)
        Yblk = torch.tensor(Msub, device=dev)             # (nTl, nTd) association targets

        hid_l = int(min(_HID_CAP, max(k, nTl)))
        hid_d = int(min(_HID_CAP, max(k, nTd)))
        self.enc_l = _VGAEEncoder(nTl, hid_l, k).to(dev)
        self.enc_d = _VGAEEncoder(nTd, hid_d, k).to(dev)

        params = list(self.enc_l.parameters()) + list(self.enc_d.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)  # paper wd=1e-5

        # class-imbalance weighting for the association BCE.
        pos = float(Msub.sum())
        neg = float(Msub.size - pos)
        pos_w = torch.tensor([neg / (pos + 1.0)], device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        mse = nn.MSELoss()

        n_outer = max(1, self.epochs // self.inner)
        self.enc_l.train(); self.enc_d.train()
        loss_val = 0.0
        for _ in range(n_outer):
            # -- E-step (LP branch): recompute the propagated label field F from a
            #    seed that folds in the VGAE's CURRENT decoded scores (R3). --------
            with torch.no_grad():
                mu_l, _ = self.enc_l(Al, Fl)
                mu_d, _ = self.enc_d(Ad, Fd)
                vg_pred = torch.sigmoid(mu_l @ mu_d.T)                # (nTl, nTd)
                seed = Yblk + self.lp_mix * vg_pred * (1.0 - Yblk)   # feedback on unobserved
                F = Pl @ seed @ Pd                                   # resolvent smoothing
                F = F / (F.max() + _EPS)                             # -> [0, 1]

            # -- M-step (VGAE branch): inner Adam steps, regularised by F (R3). ----
            for _ in range(self.inner):
                opt.zero_grad()
                mu_l, lv_l = self.enc_l(Al, Fl)
                mu_d, lv_d = self.enc_d(Ad, Fd)
                z_l = _reparam(mu_l, lv_l)
                z_d = _reparam(mu_d, lv_d)
                logits = z_l @ z_d.T                                 # R4 plain inner product
                loss = bce(logits, Yblk)
                # R1/D3: each VGAE reconstructs its OWN similarity graph.
                loss = loss + self.rec_w * (mse(torch.sigmoid(z_l @ z_l.T), SLt)
                                            + mse(torch.sigmoid(z_d @ z_d.T), SDt))
                loss = loss + self.kl_w * (_kl(mu_l, lv_l) + _kl(mu_d, lv_d))
                loss = loss + self.lp_w * mse(torch.sigmoid(logits), F)
                loss.backward()
                opt.step()
            loss_val = float(loss.item())
        self.final_loss = loss_val

        # --- final converged propagated labels for the score blend (R4) ---------
        self.enc_l.eval(); self.enc_d.eval()
        with torch.no_grad():
            mu_l, _ = self.enc_l(Al, Fl)                  # (nTl, k) train lnc
            mu_d, _ = self.enc_d(Ad, Fd)                  # (nTd, k) train dis
            vg_pred = torch.sigmoid(mu_l @ mu_d.T)
            seed = Yblk + self.lp_mix * vg_pred * (1.0 - Yblk)
            F_final = Pl @ seed @ Pd
            F_final = F_final / (F_final.max() + _EPS)    # (nTl, nTd) in [0, 1]

            # cold = zero feature on a graph-isolated node (self-loop -> A = 1).
            eye1 = torch.eye(1, device=dev)
            mu_l_cold, _ = self.enc_l(eye1, torch.zeros(1, nTl, device=dev))   # (1, k)
            mu_d_cold, _ = self.enc_d(eye1, torch.zeros(1, nTd, device=dev))   # (1, k)

            Zl = mu_l_cold.repeat(self.n_l, 1)            # (n_l, k) all-cold default
            Zl[torch.tensor(tl, device=dev, dtype=torch.long)] = mu_l
            Zd = mu_d_cold.repeat(self.n_d, 1)            # (n_d, k) all-cold default
            Zd[torch.tensor(td, device=dev, dtype=torch.long)] = mu_d

            S = torch.sigmoid(Zl @ Zd.T)                  # (n_l, n_d) latent decoder
            # R4: blend the co-trained latent score with propagated labels on the
            # warm train sub-block (cold rows/cols keep the honest latent floor).
            ti = torch.tensor(tl, device=dev, dtype=torch.long)
            tj = torch.tensor(td, device=dev, dtype=torch.long)
            blk = (1.0 - self.beta) * S[ti][:, tj] + self.beta * F_final
            S[ti.unsqueeze(1), tj.unsqueeze(0)] = blk

            assert torch.isfinite(S).all(), "VGAELDA produced non-finite scores"
            self.S = S.detach().cpu().numpy().astype(np.float32)
        self._trained = True
        return self

    def predict(self):
        return self.S


def build(device="cpu"):
    return _VGAELDA(device)
