"""Fetch human lncRNA sequences for the 240 dataset symbols from RNAcentral.

Pipeline (endpoints verified live 2026-06-26):
  name -> EBI Search   https://www.ebi.ac.uk/ebisearch/ws/rest/rnacentral?query=<NAME> AND TAXONOMY:9606
       -> top URS id (URS........_9606)
  URS  -> sequence     https://rnacentral.org/api/v1/rna/<URS_taxid>  (JSON 'sequence')

Honest coverage: symbols that don't resolve are recorded as null (no fabricated sequence).
Cache -> data/lncrna_seq.json ; report -> data/lncrna_seq_report.json
"""
import os as _os
_ROOT = _os.environ.get("CDELDA_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data"))
import os, json, time, urllib.parse, urllib.request

ROOT = _ROOT
DATA = os.environ.get("CCDIFF_DATA_DIR", os.path.join(ROOT, "data"))
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) ccdiff-research"
EBI = "https://www.ebi.ac.uk/ebisearch/ws/rest/rnacentral"
RNAC = "https://rnacentral.org/api/v1/rna/"


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def resolve_urs(name, taxid="9606"):
    q = urllib.parse.quote(f'{name} AND TAXONOMY:{taxid}')
    url = f"{EBI}?query={q}&format=json&fields=id,description&size=5"
    try:
        d = json.loads(_get(url))
    except Exception as e:
        return None, f"search_err:{type(e).__name__}"
    ents = d.get("entries", [])
    if not ents:
        return None, "no_hit"
    # prefer a hit whose description mentions the exact symbol; else first
    nm = name.upper()
    best = None
    for e in ents:
        desc = " ".join(e.get("fields", {}).get("description", [])).upper()
        if nm in desc:
            best = e["id"]; break
    return (best or ents[0]["id"]), ("exact" if best else "first")


def fetch_seq(urs_taxid):
    try:
        d = json.loads(_get(RNAC + urs_taxid))
        return d.get("sequence"), int(d.get("length") or 0)
    except Exception as e:
        return None, f"seq_err:{type(e).__name__}"


def main():
    names = open(os.path.join(DATA, "lnc_names.txt")).read().splitlines()
    shard_n = int(os.environ.get("SHARD_N", "1"))
    shard_idx = int(os.environ.get("SHARD_IDX", "0"))
    cache_path = os.path.join(DATA, os.environ.get("CCDIFF_SEQ_CACHE", "lncrna_seq.json"))
    out = json.load(open(cache_path)) if os.path.exists(cache_path) else {}   # RESUME
    seed = os.environ.get("CCDIFF_SEQ_SEED", "")                              # reuse prior resolved seqs
    if seed and os.path.exists(os.path.join(DATA, seed)):
        for k, v in json.load(open(os.path.join(DATA, seed))).items():
            out.setdefault(k, v)
    if shard_n > 1:                                                           # parallel sharding by index
        names = [n for i, n in enumerate(names) if i % shard_n == shard_idx]
    done = sum(1 for v in out.values() if v.get("seq"))
    print(f"resolving sequences for {len(names)} lncRNA symbols (human); resume: {len(out)} cached "
          f"({done} with seq)...")
    for i, name in enumerate(names):
        if name in out and (out[name].get("seq") or out[name].get("status", "").startswith("no_hit")):
            continue                                          # already resolved or confirmed-missing
        urs, how = resolve_urs(name)
        if urs is None:
            out[name] = {"urs": None, "seq": None, "length": 0, "status": how}
        else:
            seq, length = fetch_seq(urs)
            if isinstance(length, str) or seq is None:
                out[name] = {"urs": urs, "seq": None, "length": 0, "status": str(length)}
            else:
                out[name] = {"urs": urs, "seq": seq, "length": length, "status": f"ok:{how}"}
        if (i + 1) % 50 == 0:
            json.dump(out, open(cache_path, "w"))             # checkpoint
            ok = sum(1 for v in out.values() if v.get("seq"))
            print(f"  {i+1}/{len(names)}  ok={ok}", flush=True)
        time.sleep(0.05)
    n_ok = sum(1 for v in out.values() if v.get("seq"))
    n_miss = len(names) - n_ok

    json.dump(out, open(cache_path, "w"))
    lengths = [v["length"] for v in out.values() if v["seq"]]
    report = {
        "n_total": len(names), "n_resolved": n_ok, "n_missing": n_miss,
        "coverage": round(n_ok / len(names), 4),
        "seq_len": {"min": min(lengths) if lengths else 0, "max": max(lengths) if lengths else 0,
                    "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0},
        "missing_examples": [n for n, v in out.items() if not v["seq"]][:20],
    }
    json.dump(report, open(os.path.join(DATA, "lncrna_seq_report.json"), "w"), indent=2)
    print(f"\nCOVERAGE: {n_ok}/{len(names)} = {100*n_ok/len(names):.1f}%  "
          f"(missing {n_miss}) | seq len min={report['seq_len']['min']} "
          f"max={report['seq_len']['max']} mean={report['seq_len']['mean']}")
    print("missing examples:", report["missing_examples"][:12])
    print(f"Saved -> data/lncrna_seq.json, data/lncrna_seq_report.json")


if __name__ == "__main__":
    main()
