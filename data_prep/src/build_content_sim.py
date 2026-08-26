"""Build the LITERAL intrinsic-content similarity matrices used by the ORIGINAL
baselines, for each RNADisease k-core sweep variant. Leakage-safe (both depend
only on intrinsic node content, NOT on the association matrix), so held-out cold
nodes get a real, defined similarity -> this equips the content-capable baselines
with exactly the content their papers used, closing the content-blind criticism.

Per variant data_rd_l{L}d{D}/ writes:
  lnc_expr.npy       : (n_l, 54) GTEx expression, re-indexed by name from data_rd/lnc_expr.npy
                       (lncRNA expression similarity = cosine of this; Chen/Xuan-style)
  disease_semsim.npy : (n_d, n_d) Wang et al. 2007 DAG semantic similarity over the
                       Disease Ontology is_a graph (the disease semantic similarity used by
                       DSCMF/KATZLDA/VGAELDA/IPCARF/SIMCLDA in place of MeSH-DAG semantics).

Wang (2007) semantic similarity with all is_a edge weights = 0.5:
  S_A(t) = 0.5 ** shortest_is_a_distance(A -> t)     (contribution of ancestor t to term A)
  SV(A)  = sum_t S_A(t)
  sim(A,B) = sum_{t in anc(A) & anc(B)} (S_A(t)+S_B(t)) / (SV(A)+SV(B))
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json
import numpy as np
from collections import deque

ROOT = _ROOT
OBO = os.path.join(ROOT, "data_raw", "doid.obo")
SRC = os.path.join(ROOT, "data_rd")
VARIANTS = [f"l{ml}d{md}" for ml in (2, 3, 4, 5) for md in (2, 3, 4, 5)]
# a variant whose outputs already exist is skipped inside the loop (avoid clobber)
W = 0.5  # is_a semantic contribution factor (Wang 2007)


def parse_obo_dag(path):
    """Return parents[id] = set(is_a parents), alt[alt_id] = primary_id."""
    parents, alt = {}, {}
    cur, cur_alt, cur_par = None, [], []
    def flush():
        if cur:
            parents[cur] = set(cur_par)
            for a in cur_alt:
                alt[a] = cur
    for raw in open(path, errors="ignore"):
        line = raw.strip()
        if line == "[Term]":
            flush(); cur, cur_alt, cur_par = None, [], []
        elif line.startswith("id: "):
            cur = line[4:].strip()
        elif line.startswith("alt_id: "):
            cur_alt.append(line[8:].strip())
        elif line.startswith("is_a: "):
            cur_par.append(line[6:].split("!")[0].strip())
    flush()
    return parents, alt


def anc_svalues(term, parents):
    """S_A(t) = 0.5**shortest is_a distance from term up to ancestor t (incl. term=1.0)."""
    dist = {term: 0}
    q = deque([term])
    while q:
        n = q.popleft()
        for p in parents.get(n, ()):
            if p not in dist or dist[n] + 1 < dist[p]:
                dist[p] = dist[n] + 1
                q.append(p)
    return {t: W ** d for t, d in dist.items()}


def main():
    parents, alt = parse_obo_dag(OBO)
    src_lnc = open(os.path.join(SRC, "lnc_names.txt")).read().splitlines()
    src_expr = np.load(os.path.join(SRC, "lnc_expr.npy")).astype(np.float32)
    row_of = {n: i for i, n in enumerate(src_lnc)}
    Edim = src_expr.shape[1]

    for v in VARIANTS:
        d = os.path.join(ROOT, f"data_rd_{v}")
        if all(os.path.exists(os.path.join(d, f)) for f in ("lnc_expr.npy", "disease_semsim.npy")):
            print(f"[{v}] lnc_expr.npy + disease_semsim.npy exist, skip (avoid clobber)")
            continue
        lncs = open(os.path.join(d, "lnc_names.txt")).read().splitlines()
        doids = open(os.path.join(d, "disease_doids.txt")).read().splitlines()

        # expression re-index by name (zeros for lncRNAs without expression)
        expr = np.zeros((len(lncs), Edim), np.float32)
        hit = 0
        for i, n in enumerate(lncs):
            j = row_of.get(n)
            if j is not None:
                expr[i] = src_expr[j]; hit += 1
        np.save(os.path.join(d, "lnc_expr.npy"), expr)

        # disease semantic similarity (resolve alt->primary; Wang DAG)
        prim = [alt.get(x, x) for x in doids]
        sval = [anc_svalues(t, parents) for t in prim]
        SV = np.array([sum(s.values()) for s in sval], np.float64)
        n = len(doids)
        S = np.eye(n, dtype=np.float32)
        for a in range(n):
            for b in range(a + 1, n):
                sa, sb = sval[a], sval[b]
                common = sa.keys() & sb.keys()
                if common and (SV[a] + SV[b]) > 0:
                    num = sum(sa[t] + sb[t] for t in common)
                    val = num / (SV[a] + SV[b])
                    S[a, b] = S[b, a] = np.float32(val)
        np.save(os.path.join(d, "disease_semsim.npy"), S)
        off = S[~np.eye(n, dtype=bool)]
        print(f"[{v}] expr re-index {hit}/{len(lncs)} | semsim {S.shape} "
              f"off-diag mean={off.mean():.3f} max={off.max():.3f} nonzero={(off>0).mean():.2%}")


if __name__ == "__main__":
    main()
