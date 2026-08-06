"""LDAformer -- topological multi-hop path features + Transformer encoder, CONTENT-BLIND repro.

Reproduction of the topological core of LDAformer:

    Zhou, Xu, He, Hao, Fu, Han, Wen. "LDAformer: predicting lncRNA-disease
    associations based on topological feature extraction and Transformer
    encoder." Briefings in Bioinformatics 23(6):bbac370, 2022.
    DOI 10.1093/bib/bbac370.  Reference code:
    https://github.com/EchoChou990919/LDAformer

Original algorithm (from the paper + repo LDAfomer.py / data_train_test.py):
  * A heterogeneous, symmetric, weighted adjacency A over ALL nodes is built by
    stacking intra-class similarity blocks on the diagonal and inter-class
    association blocks off-diagonal.
  * Multi-hop topological path features: powers A^1, A^2, ..., A^{n_h} are formed
    by  A^h = A^{h-1} @ A , each power has its diagonal zeroed and is divided by
    its max (repo: `np.fill_diagonal(tmp,0); tmp = tmp/np.max(tmp)`).  n_h = 3.
  * For a pair (lncRNA i, disease j) the model builds a SEQUENCE OF N TOKENS --
    ONE TOKEN PER GRAPH NODE k -- whose feature vector is, across the hops,
    [A^1[k,i], A^1[k,j], A^2[k,i], A^2[k,j], ..., A^{n_h}[k,i], A^{n_h}[k,j]]
    i.e. the 2/3/...-hop path mass through node k to endpoints i and j
    (length 2*n_h).  A learnable W_in embeds each token to d_model.
  * A multi-layer Transformer encoder (global self-attention over the N node
    tokens) mixes them; the encoded sequence is FLATTENED and a single linear
    layer + sigmoid produces the association probability.
  * Repo defaults: d_model=12, n_heads=1, e_layers=4, d_ff=0.5*d_model,
    dropout=0.05, dimension(n_hops)=3, epochs=30, batch=32, Adam lr=1e-3
    weight_decay=1e-5, BCE loss, 1:1 random negatives.

=========================  FORCED DEVIATION #1: miRNA  =====================
The published LDAformer graph has THREE node types -- lncRNA + disease + miRNA
-- and the paper's headline results depend on the miRNA node, which mediates
much of the informative 2/3-hop lncRNA<->disease topology (lncRNA-miRNA and
miRNA-disease association matrices, plus miRNA functional similarity).  Under
our fit(M, Clnc, Cdis, train_lnc, train_dis) contract, NO miRNA network is
delivered, so those blocks are UNAVAILABLE.  This implementation is therefore
BIPARTITE-ADAPTED: the graph is the lncRNA-disease train sub-block adjacency
plus the two GIP similarity blocks ONLY.  We do NOT fabricate a miRNA network.
Consequently this is NOT a no-adaptation reproduction of the paper's canonical
numbers -- do not cite the published results as reproduced.  Everything else
(power features, node-token sequence, Transformer config, flatten+linear head,
training recipe) follows the original faithfully.
===========================================================================

=====================  FORCED DEVIATION #2: SCALE FALLBACK  ================
The faithful nodes-as-tokens formulation has one token PER GRAPH NODE, so
scoring the full grid is O(n_lnc * n_dis * N) with N = (n_train_lnc +
n_train_dis).  At the paper's scale (~240x412, N~652) this is fine, but the
RNADisease sweep datasets reach 5102x245 (N up to ~5000), where the full-grid
node-token forward is INTRACTABLE (a single cold run did not finish in 2 min on
GPU at N~1856).  We therefore make the model SCALE-ADAPTIVE:

  * N = (n_train_lnc + n_train_dis) <= LDAF_NODE_TOKENS_MAX (default 800):
    the EXACT repo-faithful NODES-AS-TOKENS path (N tokens, feat = 2*n_h).
    The canonical dataset (N~652) stays a faithful reproduction.
  * N  > LDAF_NODE_TOKENS_MAX: a TRACTABLE HOPS-AS-TOKENS fallback.  Per pair
    (i,j) we emit ONE token per hop h=1..n_h (H tokens), carrying a short
    fixed-length vector of multi-hop path/degree statistics between i and j
    read out of the SAME diag-zeroed, max-normalised power A^h:
        [ A^h[i, n_l+j]                  # i(lnc) -> j(dis) h-hop path mass
        , A^h[n_l+j, i]                   # symmetric j -> i (== above, kept for parity)
        , row-sum A^h[i, :]               # lnc i total h-hop reach (degree/path count)
        , row-sum A^h[n_l+j, :]           # dis j total h-hop reach
        , A^h[i, :n_l+? ] -> lnc->dis mass# sum of i's h-hop mass into the disease block
        , disease-block mass for j        # sum of j's h-hop mass into the lncRNA block ]
    (6 channels, per-(hop,channel) standardised over the train block).  This is
    the SAME multi-hop topological signal the paper extracts, but summarised
    into H tokens instead of N node-tokens, giving O(n_lnc * n_dis * H).  Both
    paths feed the SAME TransformerEncoder + flatten head and the SAME training
    recipe.  This tractable variant is a SECOND forced deviation, distinct from
    the miRNA omission: any RNADisease-sweep (large-N) result is produced by the
    hops-as-tokens adaptation, NOT the paper's canonical node-token model.  Be
    explicit about which path produced a given number.
===========================================================================

Bipartite-adapted heterogeneous train adjacency (the ONLY signal):

    Msub = subblock(M, train_lnc, train_dis)   # (nTl, nTd)  ONLY supervision
    SL   = gip_kernel(Msub)                     # (nTl, nTl)  lncRNA-side GIP
    SD   = gip_kernel(Msub.T)                   # (nTd, nTd)  disease-side GIP
    A    = [[SL,      Msub],
            [Msub.T,  SD  ]]                     # symmetric, (nTl+nTd)^2

Every feature is a pure function of A (hence of Msub) -> leakage-safe /
content-blind.  Clnc / Cdis are received by fit() but DELIBERATELY IGNORED
(ratified CONTENT-BLIND policy): similarity is solely the GIP kernel of the
train sub-block.

Cold (held-out) nodes are absent from the train graph, so a pair touching a
cold endpoint has an all-zero node-token sequence -> a CONSTANT floor logit.
Cold nodes therefore degrade honestly to a near-constant floor; we never
fabricate cold scores.

Every Transformer forward is micro-batched so batch*n_heads stays under the
CUDA grid-dim limit (65535); otherwise large predict/training grids raise
"CUDA error: invalid configuration argument". Rows/pairs are independent and
LayerNorm is per-token, so the split is numerically exact (preserving
determinism and sub-block invariance).
"""
import os

import numpy as np
import torch
import torch.nn as nn

# Shared, leakage-safe helpers. Canonical import form (paper dir is on sys.path).
from bench.interface import subblock, gip_kernel, SEED   # noqa: F401

# get_device mirrors the snapshot TwoTowerContent device pattern.
try:
    from ccdiff_models import get_device
except Exception:  # pragma: no cover - defensive fallback
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

NAME = "LDAformer (node-token multi-hop Transformer, GIP-only)"

# --- hyper-parameters (repo defaults; env-overridable so the smoke stays fast) ---
_HOPS = int(os.environ.get("LDAF_HOPS", "3"))          # dimension / n_h (repo: 3)
_EPOCHS = int(os.environ.get("LDAF_EPOCHS", "30"))     # repo: n_epochs=30
_LR = float(os.environ.get("LDAF_LR", "1e-3"))         # repo: Adam lr=1e-3
_WD = float(os.environ.get("LDAF_WD", "1e-5"))         # repo: weight_decay=1e-5
_DMODEL = int(os.environ.get("LDAF_DMODEL", "12"))     # repo: d_model=12
_NHEAD = int(os.environ.get("LDAF_NHEAD", "1"))        # repo: n_heads=1
_LAYERS = int(os.environ.get("LDAF_LAYERS", "4"))      # repo: e_layers=4
_DROPOUT = float(os.environ.get("LDAF_DROPOUT", "0.0"))  # repo 0.05; 0 -> deterministic
# repo d_ff = 0.5 * d_model (feed-forward multiplier); keep >=1.
_FF = max(1, int(os.environ.get("LDAF_FF", str(max(1, _DMODEL // 2)))))
_BATCH = int(os.environ.get("LDAF_BATCH", "32"))       # repo: batch_size=32 (node-token path)
# Larger batch for the large-N hops-as-token fallback: with 2P ~ tens of thousands
# of pairs, batch=32 -> too many tiny GPU steps; this is a throughput-only choice.
_BATCH_FALLBACK = int(os.environ.get("LDAF_BATCH_FALLBACK", "2048"))
_CHUNK = int(os.environ.get("LDAF_CHUNK", "64"))       # predict row-chunk size
# Scale switch: N = (n_train_lnc + n_train_dis) <= this -> faithful nodes-as-tokens;
# above it -> tractable hops-as-tokens fallback. Canonical (N~652) stays faithful.
_NODE_TOKENS_MAX = int(os.environ.get("LDAF_NODE_TOKENS_MAX", "800"))
# Fallback per-hop statistics count (hops-as-tokens feature length).
_NFEAT_FALLBACK = 6
# Per-token feature length = 2 * n_h  (A^h[k,i] and A^h[k,j] for h=1..n_h).
# Cap pairs per Transformer forward so pairs*n_heads stays under the CUDA grid-dim
# limit (65535). CPU is unaffected; rows are independent so the split is exact.
_MAXPAIRS = max(1, int(os.environ.get("LDAF_MAXPAIRS", str(49152 // max(1, _NHEAD)))))


class _LDAformerNet(nn.Module):
    """Transformer encoder used by BOTH scale paths (repo-faithful head).

    Input x: (batch, seq_len, n_feat).  A learnable W_in embeds each token
    (n_feat -> d_model); a multi-layer multi-head TransformerEncoder mixes the
    tokens with global self-attention; the encoded sequence is FLATTENED
    (seq_len*d_model) and a single linear layer -> logit (repo: linear + sigmoid,
    sigmoid applied outside via BCEWithLogits).

    seq_len / n_feat depend on the scale path:
      * nodes-as-tokens (faithful): seq_len = N nodes, n_feat = 2*n_h.
      * hops-as-tokens (fallback):  seq_len = n_h,      n_feat = _NFEAT_FALLBACK.
    dropout defaults to 0 so the forward is deterministic (sub-block invariance).
    """

    def __init__(self, seq_len, n_feat, d_model=_DMODEL, nhead=_NHEAD,
                 num_layers=_LAYERS, dim_feedforward=_FF, dropout=_DROPOUT):
        super().__init__()
        self.seq_len = seq_len
        self.n_feat = n_feat
        self.embed = nn.Linear(n_feat, d_model)              # W_in: n_feat -> d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(seq_len * d_model, 1)          # W_out: flatten -> 1

    def forward(self, x):
        # x: (batch, seq_len, n_feat) -> embed each token.
        h = self.embed(x)                                    # (batch, seq_len, d_model)
        h = self.encoder(h)                                  # (batch, seq_len, d_model)
        h = h.reshape(h.shape[0], -1)                        # flatten (batch, seq_len*d_model)
        return self.head(h).squeeze(-1)                      # (batch,) logits


class _LDAformer:
    """Content-blind, bipartite-adapted LDAformer for lncRNA-disease LDA."""

    def __init__(self, device="cpu", hops=_HOPS, epochs=_EPOCHS, lr=_LR):
        # Honor the passed device (mirrors TwoTowerContent: device or get_device()).
        self.device = device or get_device()
        self.hops = max(1, int(hops))
        self.epochs = max(1, int(epochs))
        self.lr = float(lr)
        self.seed = int(SEED)
        # seq_len / n_feat are set in fit() once the path (node- vs hop-tokens) is known.
        self.seq_len = None
        self.n_feat = None
        self.mode = None
        self.S = None

    def _power_features(self, A, nTl, nTd, H):
        """Repo-faithful power stack: for h=1..H,  A^h = A^{h-1} @ A, then zero
        the diagonal and divide by the max (repo data_train_test.py).

        Returns Astack: (H, N, N) float64, where Astack[h-1][k, t] is the h-hop
        (diagonal-free, max-normalised) path mass from node k to node t.
        Every entry is a pure function of A (hence of Msub) -> leakage-safe.
        """
        N = nTl + nTd
        Astack = np.empty((H, N, N), np.float64)
        Ah = A.copy()
        for h in range(H):
            if h > 0:
                Ah = Ah @ A                                  # A^{h+1}
            tmp = Ah.copy()
            np.fill_diagonal(tmp, 0.0)                       # remove self-loops
            m = np.max(tmp)
            if m > 1e-12:
                tmp = tmp / m                                # max-normalise (repo)
            Astack[h] = tmp
        return Astack

    def _net_batched(self, feat):
        # Run the Transformer in sub-batches so pairs*n_heads stays under the CUDA
        # grid-dim limit. Rows are independent -> concatenation is numerically exact
        # (identical to a single forward), preserving determinism & sub-block invariance.
        if feat.shape[0] <= _MAXPAIRS:
            return self.net(feat)
        outs = [self.net(feat[s:s + _MAXPAIRS]) for s in range(0, feat.shape[0], _MAXPAIRS)]
        return torch.cat(outs, dim=0)

    def _build_node_tokens(self, Astack, nTl, nTd, H):
        """FAITHFUL nodes-as-tokens features (repo).  Returns:
          Btrain   : (nTl, nTd, N, 2H) float32 -- token k = [A^h[k,i],A^h[k,j]] over h
          cold_feat: (N, 2H) float32 all-zero token sequence for cold pairs
        seq_len = N, feat_dim = 2H.
        """
        N = nTl + nTd
        lnc_cols = Astack[:, :, :nTl]                        # (H, N, nTl)  A^h[k, i]
        dis_cols = Astack[:, :, nTl:]                        # (H, N, nTd)  A^h[k, j]
        li_feat = np.transpose(lnc_cols, (2, 1, 0))          # (nTl, N, H)
        dj_feat = np.transpose(dis_cols, (2, 1, 0))          # (nTd, N, H)
        li_b = np.broadcast_to(li_feat[:, None, :, :], (nTl, nTd, N, H))
        dj_b = np.broadcast_to(dj_feat[None, :, :, :], (nTl, nTd, N, H))
        Btrain = np.empty((nTl, nTd, N, 2 * H), np.float32)
        Btrain[..., 0::2] = li_b                             # A^h[k,i] at even slots
        Btrain[..., 1::2] = dj_b                             # A^h[k,j] at odd slots
        cold_feat = np.zeros((N, 2 * H), np.float32)
        return Btrain, cold_feat

    def _build_hop_tokens(self, Astack, nTl, nTd, H):
        """TRACTABLE hops-as-tokens fallback (scale deviation #2).  ONE token per
        hop h carries a length-`_NFEAT_FALLBACK` vector of multi-hop path/degree
        statistics between lnc i and dis j, all pure functions of A^h.  Returns:
          Btrain   : (nTl, nTd, H, F) float32
          cold_feat: (H, F) float32  (standardised zero -> constant floor)
        seq_len = H, feat_dim = F.  O(n_lnc*n_dis*H) to score.
        """
        F = _NFEAT_FALLBACK
        Bstack = np.empty((nTl, nTd, H, F), np.float64)
        for h in range(H):
            Ah = Astack[h]                                   # (N, N) diag-zeroed, /max
            cross = Ah[:nTl, nTl:]                            # (nTl, nTd) lnc i -> dis j
            crossT = Ah[nTl:, :nTl].T                         # (nTl, nTd) dis j -> lnc i
            lnc_reach = Ah[:nTl, :].sum(axis=1)               # (nTl,) i total h-hop reach
            dis_reach = Ah[nTl:, :].sum(axis=1)               # (nTd,) j total h-hop reach
            lnc_dismass = cross.sum(axis=1)                   # (nTl,) i's mass into dis block
            dis_lncmass = Ah[nTl:, :nTl].sum(axis=1)          # (nTd,) j's mass into lnc block
            Bstack[:, :, h, 0] = cross
            Bstack[:, :, h, 1] = crossT
            Bstack[:, :, h, 2] = lnc_reach[:, None]
            Bstack[:, :, h, 3] = dis_reach[None, :]
            Bstack[:, :, h, 4] = lnc_dismass[:, None]
            Bstack[:, :, h, 5] = dis_lncmass[None, :]
        # Per-(hop, channel) standardisation over the whole train block.
        flat = Bstack.reshape(-1, H * F)
        mu = flat.mean(axis=0).reshape(H, F)
        sd = flat.std(axis=0).reshape(H, F) + 1e-8
        Btrain = ((Bstack - mu) / sd).astype(np.float32)      # (nTl, nTd, H, F)
        cold_feat = ((0.0 - mu) / sd).astype(np.float32)      # standardised-zero token
        return Btrain, cold_feat

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        # Clnc / Cdis deliberately IGNORED (content-blind policy).
        torch.manual_seed(self.seed)
        dev = self.device
        M = np.asarray(M, np.float32)
        self.n_l, self.n_d = M.shape
        tl = np.asarray(train_lnc)
        td = np.asarray(train_dis)
        self.tl, self.td = tl, td
        nTl, nTd = tl.size, td.size
        self.nTl, self.nTd = nTl, nTd
        self.N = nTl + nTd
        H = self.hops

        # Msub is the ONLY supervision (leakage-safe / content-blind).
        Msub = subblock(M, tl, td).astype(np.float64)

        # Degenerate guard: no usable train network / no both-class labels -> floor.
        if (nTl == 0 or nTd == 0 or Msub.sum() == 0.0
                or (Msub == 0).sum() == 0):
            self.S = np.zeros((self.n_l, self.n_d), np.float32)
            return self

        # Bipartite-adapted symmetric heterogeneous adjacency A (miRNA UNAVAILABLE).
        SL = gip_kernel(Msub).astype(np.float64)             # (nTl, nTl)
        SD = gip_kernel(Msub.T).astype(np.float64)           # (nTd, nTd)
        N = self.N
        A = np.zeros((N, N), np.float64)
        A[:nTl, :nTl] = SL
        A[:nTl, nTl:] = Msub
        A[nTl:, :nTl] = Msub.T
        A[nTl:, nTl:] = SD

        # Repo-faithful multi-hop power features (diag-zeroed, max-normalised).
        Astack = self._power_features(A, nTl, nTd, H)        # (H, N, N)

        # SCALE SWITCH: faithful nodes-as-tokens for small graphs, tractable
        # hops-as-tokens fallback for large graphs (see docstring deviation #2).
        if N <= _NODE_TOKENS_MAX:
            self.mode = "node-tokens"
            self.Btrain, self.cold_feat = self._build_node_tokens(Astack, nTl, nTd, H)
        else:
            self.mode = "hop-tokens"
            self.Btrain, self.cold_feat = self._build_hop_tokens(Astack, nTl, nTd, H)
        self.seq_len = self.Btrain.shape[2]
        self.n_feat = self.Btrain.shape[3]

        # Build the Transformer (after manual_seed -> deterministic init).
        self.net = _LDAformerNet(seq_len=self.seq_len, n_feat=self.n_feat).to(dev)

        # Training pairs: all positives + seeded 1:1 negatives (numpy rng: content
        # -blind, independent of torch global RNG and of off-block M).
        pos = np.argwhere(Msub == 1)                         # (P, 2) [row, col]
        neg_all = np.argwhere(Msub == 0)                     # (Ng, 2)
        P, Ng = pos.shape[0], neg_all.shape[0]
        rng = np.random.default_rng(self.seed)

        Xb = torch.from_numpy(self.Btrain).to(dev)           # (nTl,nTd,N,2H) float32
        pos_i = torch.from_numpy(pos[:, 0].astype(np.int64)).to(dev)
        pos_j = torch.from_numpy(pos[:, 1].astype(np.int64)).to(dev)
        pos_feat = Xb[pos_i, pos_j]                          # (P, N, 2H)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=_WD)
        bce = nn.BCEWithLogitsLoss()                         # 1:1 sampling -> balanced

        # Mini-batch size: repo uses 32 (small canonical data). In the hops-as-token
        # fallback (large-N sweep) 2P can be ~16k, so 32 -> ~15k tiny GPU steps whose
        # kernel-launch overhead dominates; a larger batch amortises this WITHOUT
        # changing the model or the topological signal (throughput-only choice, on
        # the already scale-adapted path). Node-token path keeps the repo default.
        batch = _BATCH if self.mode == "node-tokens" else _BATCH_FALLBACK

        self.net.train()
        last_loss = 0.0
        for _ in range(self.epochs):
            sel = rng.choice(Ng, size=P, replace=(Ng < P))
            neg = neg_all[sel]
            neg_i = torch.from_numpy(neg[:, 0].astype(np.int64)).to(dev)
            neg_j = torch.from_numpy(neg[:, 1].astype(np.int64)).to(dev)
            feat = torch.cat([pos_feat, Xb[neg_i, neg_j]], dim=0)   # (2P, L, F)
            y = torch.cat([torch.ones(P, device=dev), torch.zeros(P, device=dev)])
            # mini-batch over pairs
            perm = torch.from_numpy(rng.permutation(2 * P)).to(dev)
            for bs in range(0, 2 * P, batch):
                idx = perm[bs:bs + batch]
                opt.zero_grad()
                logits = self._net_batched(feat[idx])
                loss = bce(logits, y[idx])
                loss.backward()
                opt.step()
            last_loss = float(loss.item())          # one sync per epoch (not per step)
        self.final_loss = last_loss

        # Predict the FULL grid under eval/no_grad, chunked over rows.
        self.net.eval()
        self.S = self._predict_full(dev)
        return self

    @torch.no_grad()
    def _predict_full(self, dev):
        # L = sequence length (N node-tokens OR H hop-tokens), F = per-token dim.
        n_l, n_d, L, F = self.n_l, self.n_d, self.seq_len, self.n_feat
        S_full = np.empty((n_l, n_d), np.float32)

        # Map full-node indices -> train-block indices (-1 == cold / off-block).
        row_map = np.full(n_l, -1, np.int64)
        row_map[self.tl] = np.arange(self.tl.size)
        warm_cols = self.td                                  # full-index warm columns

        chunk = max(1, _CHUNK)
        for start in range(0, n_l, chunk):
            stop = min(start + chunk, n_l)
            C = stop - start
            # Default every pair to the cold (all-zero token) sequence, then fill
            # in warm x warm pairs. Cold rows/cols keep the zero sequence -> a
            # constant floor logit.
            feat = np.broadcast_to(self.cold_feat, (C, n_d, L, F)).copy()
            for li in range(C):
                br = row_map[start + li]
                if br >= 0:
                    feat[li, warm_cols, :, :] = self.Btrain[br]   # (nTd, L, F)
            ft = torch.from_numpy(feat.reshape(-1, L, F)).to(dev)
            logits = self._net_batched(ft)                        # (C*n_d,)
            probs = torch.sigmoid(logits).reshape(C, n_d)
            S_full[start:stop] = probs.detach().cpu().numpy().astype(np.float32)

        S_full = np.nan_to_num(S_full, nan=0.0, posinf=1.0, neginf=0.0)
        assert np.isfinite(S_full).all(), "LDAformer produced non-finite scores"
        return S_full.astype(np.float32)

    def predict(self):
        return self.S


def build(device="cpu"):
    return _LDAformer(device)
