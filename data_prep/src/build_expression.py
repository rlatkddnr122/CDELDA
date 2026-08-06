"""lncRNA EXPRESSION content view (orthogonal to sequence). GTEx v8 gene median TPM across 54 tissues.
Maps our lncRNA symbols -> GTEx 54-d log1p(TPM) vector, L2-normalized. Intrinsic -> defined for cold.

Symbol matching (exact only, NO fuzzy / NO clone-name suffix strip -> avoids fabricated matches):
  1. direct: name == GTEx Description (symbol)
  2. HGNC-resolved: name (approved/alias/prev symbol via hgnc_complete_set.txt) -> approved symbol
     matched on GTEx Description, OR its ensembl_gene_id matched on GTEx Name (version-stripped).
HGNC is the official human gene-naming authority -> alias resolution is authoritative, not heuristic.
If hgnc_complete_set.txt is absent, falls back to direct-symbol only (original behavior).
Saves CCDIFF_DATA_DIR/lnc_expr.npy. Reports coverage + per-route breakdown honestly.
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json, gzip, csv, urllib.request
import numpy as np

DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(_ROOT, "data"))
RAW = os.path.join(_ROOT, "data_raw")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)"
URL = ("https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
       "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz")
HGNC = os.path.join(RAW, "hgnc_complete_set.txt")


def load_hgnc():
    """upper-name -> approved symbol (alias/prev) ; approved symbol(upper) -> ensembl_gene_id."""
    sym2ens, alias2sym = {}, {}
    if not os.path.exists(HGNC):
        return sym2ens, alias2sym
    with open(HGNC) as f:
        r = csv.reader(f, delimiter="\t"); H = next(r); ci = {h: i for i, h in enumerate(H)}
        cs, cal, cpv, cen = ci["symbol"], ci["alias_symbol"], ci["prev_symbol"], ci["ensembl_gene_id"]
        for row in r:
            s = row[cs].strip()
            if not s:
                continue
            sym2ens[s.upper()] = row[cen].strip()
            for fld in (cal, cpv):                      # alias_symbol/prev_symbol are '|'-separated
                for a in row[fld].split("|"):
                    a = a.strip()
                    if a:
                        alias2sym.setdefault(a.upper(), s.upper())
    return sym2ens, alias2sym


def main():
    gz = os.path.join(RAW, "gtex_gene_median_tpm.gct.gz")
    if not os.path.exists(gz):
        req = urllib.request.Request(URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=180) as r, open(gz, "wb") as f:
            f.write(r.read())
    sym2vec, ens2vec = {}, {}
    with gzip.open(gz, "rt") as f:
        f.readline(); f.readline()                    # #1.2 ; nrows ncols
        header = f.readline().rstrip("\n").split("\t")
        n_tissue = len(header) - 2
        for line in f:
            c = line.rstrip("\n").split("\t")
            ens, sym = c[0].split(".")[0], c[1]
            vec = np.array(c[2:], dtype=np.float32)
            sym2vec.setdefault(sym.upper(), vec)       # first occurrence
            ens2vec.setdefault(ens, vec)
    sym2ens, alias2sym = load_hgnc()
    names = open(os.path.join(DATA, "lnc_names.txt")).read().splitlines()

    def lookup(nm):
        u = nm.upper()
        if u in sym2vec:
            return sym2vec[u], "direct-sym"
        e = sym2ens.get(u)                              # name is an approved symbol -> its ensembl
        if e and e in ens2vec:
            return ens2vec[e], "approved-ens"
        a = alias2sym.get(u)                            # name is alias/prev -> approved symbol
        if a:
            if a in sym2vec:
                return sym2vec[a], "alias-sym"
            e2 = sym2ens.get(a)
            if e2 and e2 in ens2vec:
                return ens2vec[e2], "alias-ens"
        return None, None

    from collections import Counter
    X, n_ok, route = [], 0, Counter()
    for nm in names:
        v, how = lookup(nm)
        if v is not None:
            X.append(np.log1p(v)); n_ok += 1; route[how] += 1
        else:
            X.append(np.zeros(n_tissue, np.float32))
    X = np.stack(X).astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    hgnc_on = os.path.exists(HGNC)
    print(f"lnc expression: {X.shape} ({n_tissue} tissues) | matched {n_ok}/{len(names)} "
          f"({100*n_ok/len(names):.1f}%) | HGNC={'on' if hgnc_on else 'off'} | routes={dict(route)}")
    np.save(os.path.join(DATA, "lnc_expr.npy"), X)
    json.dump({"dim": int(X.shape[1]), "coverage": n_ok, "n": len(names),
               "source": "GTEx v8 gene median TPM, 54 tissues, log1p",
               "matching": "direct-symbol + HGNC alias/prev + Ensembl-id (exact only)" if hgnc_on
                           else "direct-symbol only",
               "routes": dict(route)},
              open(os.path.join(DATA, "lnc_expr_meta.json"), "w"), indent=2)
    print(f"Saved -> {DATA}/lnc_expr.npy")


if __name__ == "__main__":
    main()
