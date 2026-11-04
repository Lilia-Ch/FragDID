from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import FilterCatalog

RDLogger.DisableLog("rdApp.*")

try:
    import sys
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer                                         
    HAVE_SA = True
except Exception:                                       
    HAVE_SA = False


def _catalog(*names) -> FilterCatalog.FilterCatalog:
    p = FilterCatalog.FilterCatalogParams()
    for n in names:
        p.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, n))
    return FilterCatalog.FilterCatalog(p)


PAINS = _catalog("PAINS_A", "PAINS_B", "PAINS_C")
BRENK = _catalog("BRENK")
_REF_CACHE: Dict[tuple, tuple] = {}


def canonical_set(reference: Sequence[str]) -> frozenset:
    key = (id(reference), len(reference))
    hit = _REF_CACHE.get(key)
    if hit is not None and hit[0] is reference:
        return hit[1]
    out = frozenset(
        Chem.MolToSmiles(m) for m in
        (Chem.MolFromSmiles(s) for s in reference) if m is not None)
    if len(_REF_CACHE) >= 4:
        _REF_CACHE.clear()
    _REF_CACHE[key] = (reference, out)
    return out
def describe(mol: Chem.Mol) -> Dict[str, float]:
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    heavy = mol.GetNumHeavyAtoms()
    non_ch = sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H"))
    d = {
        "logp": Crippen.MolLogP(mol),
        "mw": Descriptors.MolWt(mol),
        "tpsa": Descriptors.TPSA(mol),
        "qed": QED.qed(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotb": Lipinski.NumRotatableBonds(mol),
        "heavy": heavy,
        "n_rings": len(rings),
        "n_aromatic_rings": sum(1 for r in rings
                                if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)),
        "max_ring": max((len(r) for r in rings), default=0),
        "r_value": (non_ch / heavy) if heavy else 0.0,
        "pains": int(PAINS.HasMatch(mol)),
        "brenk": int(BRENK.HasMatch(mol)),
    }
    d["sa"] = sascorer.calculateScore(mol) if HAVE_SA else float("nan")
    d["sa_norm"] = (10.0 - d["sa"]) / 9.0 if HAVE_SA else float("nan")
    d["lipinski"] = int((d["mw"] <= 500) + (d["logp"] <= 5)
                        + (d["hbd"] <= 5) + (d["hba"] <= 10))
    return d

def _fps(mols: Sequence[Chem.Mol]):
    return [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]


def diversity(mols: Sequence[Chem.Mol], cap: int = 1000) -> float:
    fps = _fps(mols[:cap])
    if len(fps) < 2:
        return float("nan")
    tot = n = 0.0
    for i in range(len(fps)):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        tot += sum(s)
        n += len(s)
    return 1.0 - tot / max(n, 1)


def n_circles(mols: Sequence[Chem.Mol], thresh: float = 0.75, cap: int = 1000) -> int:
    picked = []
    for fp in _fps(mols[:cap]):
        if not picked or max(DataStructs.BulkTanimotoSimilarity(fp, picked)) < thresh:
            picked.append(fp)
    return len(picked)


def wasserstein(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


CORE_ORDER = [
    "success_oracle", "success_eval", "success_proxy","mae_to_window", "proxy_error", "validity", "uniqueness", "novelty","diversity", "n_circles", "n_scaffolds","sa_mean", "frac_sa_le_4.5", "qed_mean","frac_pains_free", "frac_brenk_free", "quality_pass","n_generated", "n_decoder_calls", "seconds", "seconds_per_hit",
]

GROUPS = [
    ("target",       ["success_oracle", "success_eval", "success_proxy","mae_to_window", "proxy_error"]),
    ("set quality",  ["validity", "uniqueness", "novelty"]),
    ("diversity",    ["diversity", "n_circles", "n_scaffolds"]),
    ("chemistry",    ["sa_mean", "frac_sa_le_4.5", "qed_mean","frac_pains_free", "frac_brenk_free", "quality_pass"]),
    ("cost",         ["n_generated", "n_decoder_calls", "seconds","seconds_per_hit"]),
]

LABEL = {
    "success_oracle": "success (oracle)",
    "success_eval":   "success (independent evaluator)",
    "success_proxy":  "success (proxy that drove the search)",
    "mae_to_window":  "mean distance to the window",
    "proxy_error":    "proxy error on generated molecules",
    "validity":       "valid",
    "uniqueness":     "unique",
    "novelty":        "not in the reference set",
    "diversity":      "mean pairwise Tanimoto distance",
    "n_circles":      "distinct chemotypes (#circles, 0.75)",
    "n_scaffolds":    "Murcko scaffolds",
    "sa_mean":        "SA score (1 easy, 10 hard)",
    "frac_sa_le_4.5": "fraction with SA <= 4.5",
    "qed_mean":       "QED",
    "frac_pains_free": "PAINS-free",
    "frac_brenk_free": "Brenk-free",
    "quality_pass":   "pass SA + PAINS + ring + MW/HBD/HBA",
    "n_generated":    "molecules reported",
    "n_decoder_calls": "decoder calls",
    "seconds":        "seconds",
    "seconds_per_hit": "seconds per in-window molecule",
}


def evaluate(smiles: Sequence[str],
             reference: Optional[Sequence[str]] = None,
             target: Optional[tuple] = None,
             pred: Optional[Sequence[float]] = None,
             oracle: str = "logp",
             extern: Optional[Sequence[float]] = None,
             full: bool = False) -> Dict[str, object]:
    n_in = len(smiles)
    mols, canon, keep_idx = [], [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if s else None
        if m is None or m.GetNumHeavyAtoms() == 0:
            continue
        mols.append(m)
        canon.append(Chem.MolToSmiles(m))
        keep_idx.append(i)

    out: Dict[str, object] = {
        "n_generated": n_in,
        "validity": len(mols) / max(n_in, 1),
        "uniqueness": len(set(canon)) / max(len(canon), 1),
    }
    if not mols:
        return out

    D = [describe(m) for m in mols]
    col = lambda k: np.array([d[k] for d in D], float)

    if reference is not None:
        ref = canonical_set(reference)
        out["novelty"] = sum(c not in ref for c in canon) / len(canon)
    in_target = None
    if target is not None:
        lo, hi = target
        if oracle in D[0]:
            v = col(oracle)
            in_target = (v >= lo) & (v <= hi)
            out["success_oracle"] = float(np.mean(in_target))
            out["n_in_window"] = int(in_target.sum())
            out["mae_to_window"] = float(np.mean(
                np.maximum(0.0, np.maximum(lo - v, v - hi))))
        if pred is not None:
            p = np.array([pred[i] for i in keep_idx], float)
            out["success_proxy"] = float(np.mean((p >= lo) & (p <= hi)))
            if in_target is not None:
                out["proxy_error"] = float(np.mean(np.abs(p - col(oracle))))
            elif "mae_to_window" not in out:
                out["mae_to_window"] = float(np.mean(
                    np.maximum(0.0, np.maximum(lo - p, p - hi))))
        if extern is not None:
            e = np.array([extern[i] for i in keep_idx], float)
            ok = np.isfinite(e)
            if ok.any():
                ein = (e[ok] >= lo) & (e[ok] <= hi)
                out["success_eval"] = float(np.mean(ein))
                out["n_in_window_eval"] = int(ein.sum())
                if pred is not None:
                    p = np.array([pred[i] for i in keep_idx], float)
                    out["proxy_error"] = float(np.mean(np.abs(p[ok] - e[ok])))
                if in_target is None:
                    out["mae_to_window"] = float(np.mean(
                        np.maximum(0.0, np.maximum(lo - e[ok], e[ok] - hi))))
                    in_target = np.zeros(len(mols), bool)
                    in_target[np.where(ok)[0]] = ein
    out["sa_mean"] = float(np.nanmean(col("sa")))
    out["frac_sa_le_4.5"] = float(np.mean(col("sa") <= 4.5))
    out["qed_mean"] = float(np.nanmean(col("qed")))
    out["frac_pains_free"] = float(np.mean(col("pains") == 0))
    out["frac_brenk_free"] = float(np.mean(col("brenk") == 0))

    ok = ((col("sa") <= 4.5) & (col("pains") == 0) & (col("max_ring") <= 7)
          & (col("mw") <= 500) & (col("hbd") <= 5) & (col("hba") <= 10))
    out["quality_pass"] = float(np.mean(ok))
    if in_target is not None:
        out["quality_pass_in_window"] = float(np.mean(ok & in_target))

    out["diversity"] = diversity(mols)
    out["n_circles"] = n_circles(mols)
    if in_target is not None and int(in_target.sum()) >= 2:
        out["diversity_in_window"] = diversity(
            [m for m, t in zip(mols, in_target) if t])
    scaf = Counter()
    for m in mols:
        try:
            scaf[MurckoScaffold.MurckoScaffoldSmiles(mol=m)] += 1
        except Exception:
            pass
    out["n_scaffolds"] = len(scaf)

    if not full:
        return out

    for k in ("mw", "tpsa", "logp", "heavy", "rotb", "n_rings","n_aromatic_rings", "r_value"):
        out[f"{k}_mean"] = float(np.nanmean(col(k)))
    out["quality_pass_strict"] = float(np.mean(
        ok & (col("brenk") == 0) & (col("r_value") >= 0.05)
        & (col("r_value") <= 0.50)))
    out["lipinski_all4"] = float(np.mean(col("lipinski") == 4))
    out["scaffold_ratio"] = len(scaf) / max(len(mols), 1)
    if reference is not None:
        rmols = [m for m in (Chem.MolFromSmiles(s) for s in reference[:5000])
                 if m is not None]
        if rmols:
            R = [describe(m) for m in rmols]
            for k in ("mw", "logp", "tpsa", "sa", "n_rings", "qed"):
                out[f"w1_{k}"] = wasserstein(col(k), [r[k] for r in R])
    return out


def read_smiles_file(path: str, col: Optional[str] = None,
                     pred_col: Optional[str] = None):
    smis, preds = [], []
    if path.lower().endswith((".csv", ".tsv")):
        delim = "\t" if path.lower().endswith(".tsv") else ","
        with open(path, newline="") as f:
            rd = csv.DictReader(f, delimiter=delim)
            names = rd.fieldnames or []
            c = col or next((x for x in ("smiles", "SMILES", "smile", "mol")
                             if x in names), names[0])
            for row in rd:
                smis.append((row.get(c) or "").strip())
                if pred_col and pred_col in row:
                    try:
                        preds.append(float(row[pred_col]))
                    except (TypeError, ValueError):
                        preds.append(float("nan"))
    else:
        with open(path) as f:
            smis = [ln.split()[0] for ln in f if ln.strip()]
    return smis, (preds or None)


def print_stats(res: Dict[str, object]) -> None:
    NEEDS = {"novelty": "pass --reference", "success_oracle": "pass --oracle logp|qed","success_eval": "pass --eval-model"}
    shown = set()
    for title, keys in GROUPS:
        present = [k for k in keys if k in res or k in NEEDS]
        if not present:
            continue
        print(f"\n  {title}")
        for k in present:
            shown.add(k)
            lab = LABEL.get(k, k)
            if k not in res:
                print(f"    {lab:<42s} not computed ({NEEDS[k]})")
                continue
            v = res[k]
            print(f"    {lab:<42s} {v:.4f}" if isinstance(v, float)
                  else f"    {lab:<42s} {v}")
    extra = [k for k in res if k not in shown and not k.startswith("w1_")
             and not k.endswith("_mean")]
    if extra:
        print("\n  also")
        for k in sorted(extra):
            v = res[k]
            print(f"    {k:<42s} {v:.4f}" if isinstance(v, float)
                  else f"    {k:<42s} {v}")


def main(argv=None):
    p = argparse.ArgumentParser("fraginv.metrics")
    p.add_argument("--smiles", required=True, help="CSV or TXT of generated molecules")
    p.add_argument("--column", default=None)
    p.add_argument("--pred-col", default=None,
                   help="column holding the proxy prediction, e.g. 'pred'")
    p.add_argument("--eval-col", default=None,
                   help="column holding an independent evaluator's prediction")
    p.add_argument("--reference", default=None, help="training set, for novelty/W1")
    p.add_argument("--lo", type=float, default=None)
    p.add_argument("--hi", type=float, default=None)
    p.add_argument("--oracle", default="logp",
                   choices=["logp", "qed", "sa", "tpsa", "mw", "none"])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    smis, preds = read_smiles_file(a.smiles, a.column, a.pred_col)
    extern = read_smiles_file(a.smiles, a.column, a.eval_col)[1] if a.eval_col else None
    ref = read_smiles_file(a.reference)[0] if a.reference else None
    tgt = (a.lo, a.hi) if a.lo is not None and a.hi is not None else None
    res = evaluate(smis, ref, tgt, preds, a.oracle, extern)

    print_stats(res)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
