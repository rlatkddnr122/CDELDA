"""Two-sided (lncRNA seq-kmer  +  disease text) cold-start — common utilities.

Three inductive scenarios, all via one fit(M, Clnc, Cdis, train_lnc, train_dis):
  disease-cold : train_lnc=ALL, train_dis=train   -> eval (all lnc  x cold dis)
  lncRNA-cold  : train_lnc=train, train_dis=ALL    -> eval (cold lnc x all dis)
  both-cold C4 : train_lnc=train, train_dis=train   -> eval (cold lnc x cold dis)  [neither seen]

A model trains ONLY on the M[train_lnc][:, train_dis] sub-block (held-out nodes have zero observed
associations) and must use content towers to score cold rows/cols.
"""
import os, numpy as np
from ccdiff_common import recall_at_k, auc_aupr_global, auc_aupr_sampled, personalization, SEED

DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def load():
    """lncRNA feature file is env-switchable (CCDIFF_LNC_FILE): 'lnc_kmer.npy' (default, k-mer)
    or 'lnc_rnafm.npy' (RNA-FM encoder)."""
    lnc_file = os.environ.get("CCDIFF_LNC_FILE", "lnc_kmer.npy")
    dis_file = os.environ.get("CCDIFF_DIS_FILE", "disease_emb.npy")
    M = np.load(os.path.join(DATA, "M.npy")).astype(np.float32)            # (240,412)
    Clnc = np.load(os.path.join(DATA, lnc_file)).astype(np.float32)        # (240, d_l)
    Cdis = np.load(os.path.join(DATA, dis_file)).astype(np.float32)        # (n_dis, d_d)
    assert Clnc.shape[0] == M.shape[0] and Cdis.shape[0] == M.shape[1], \
        f"shape mismatch: M{M.shape} Clnc{Clnc.shape} Cdis{Cdis.shape}"
    return M, Clnc, Cdis


def folds(n, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    return [np.sort(f) for f in np.array_split(rng.permutation(n), k)]


def scenario_indices(scn, lnc_fold, dis_fold, n_lnc=240, n_dis=412):
    """Return (train_lnc, train_dis, eval_lnc, eval_dis) for a scenario and a paired fold."""
    all_l, all_d = np.arange(n_lnc), np.arange(n_dis)
    tr_l = np.setdiff1d(all_l, lnc_fold); tr_d = np.setdiff1d(all_d, dis_fold)
    if scn == "disease":
        return all_l, tr_d, all_l, dis_fold            # all lnc seen; cold diseases
    if scn == "lncRNA":
        return tr_l, all_d, lnc_fold, all_d            # all dis seen; cold lncRNAs
    if scn == "both":
        return tr_l, tr_d, lnc_fold, dis_fold          # both held out (true C4 block)
    raise ValueError(scn)


def eval_block(S_full, M_full, eval_lnc, eval_dis, query_axis, seed=SEED):
    """Evaluate the eval block. query_axis decides ranking orientation:
       'disease' -> per cold disease, rank lncRNAs ; 'lncRNA' -> per cold lncRNA, rank diseases."""
    S = S_full[np.ix_(eval_lnc, eval_dis)]      # (|eval_lnc|, |eval_dis|)
    Y = M_full[np.ix_(eval_lnc, eval_dis)]
    if query_axis == "disease":                 # queries = diseases (cols) -> rank lncRNAs (rows)
        S, Y = S.T, Y.T                         # (n_query_dis, n_cand_lnc)
    # else queries = lncRNAs (rows) -> rank diseases (cols): keep as is
    auc_all, aupr_all = auc_aupr_global(S, Y)
    auc_11, aupr_11 = auc_aupr_sampled(S, Y, ratio=1, seed=seed)
    res = {"AUC_all": auc_all, "AUPR_all": aupr_all, "AUC_1to1": auc_11, "AUPR_1to1": aupr_11,
           "personalization": personalization(S), "n_pos": int(Y.sum()),
           "n_query": int(Y.shape[0]), "n_cand": int(Y.shape[1])}
    res.update(recall_at_k(S, Y, ks=(10, 20, 50)))
    return res
