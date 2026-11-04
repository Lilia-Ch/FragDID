from __future__ import annotations

import csv
import os
import urllib.request
from typing import List, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HARTREE_TO_EV = 27.211386245988
QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"


def read_smiles(path: str, column: Optional[str] = None,
                limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    if path.lower().endswith((".csv", ".tsv")):
        delim = "\t" if path.lower().endswith(".tsv") else ","
        with open(path, newline="") as f:
            rd = csv.DictReader(f, delimiter=delim)
            names = rd.fieldnames or []
            c = column or next((x for x in ("smiles", "SMILES", "smile", "mol")
                                if x in names), names[0])
            for row in rd:
                s = (row.get(c) or "").strip()
                if s:
                    out.append(s)
                if limit and len(out) >= limit:
                    break
    else:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    out.append(ln.split()[0])
                if limit and len(out) >= limit:
                    break
    return out


def read_targets(path: str, column: str,
                 limit: Optional[int] = None) -> List[float]:
    vals: List[float] = []
    delim = "\t" if path.lower().endswith(".tsv") else ","
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=delim):
            try:
                vals.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                vals.append(float("nan"))
            if limit and len(vals) >= limit:
                break
    return vals

def prep_qm9(out_csv: str, source: Optional[str] = None,
             url: str = QM9_URL, limit: Optional[int] = None) -> str:
    if source is None:
        cache = os.path.join(os.path.dirname(out_csv) or ".", "qm9_raw.csv")
        if not os.path.exists(cache):
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            print(f"downloading QM9 from {url}")
            urllib.request.urlretrieve(url, cache)
        source = cache

    with open(source, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{source} is empty")

    names = rows[0].keys()
    smi_col = next((c for c in ("smiles", "SMILES", "mol_smiles") if c in names), None)
    gap_col = next((c for c in ("gap", "gap_ev", "homo_lumo_gap") if c in names), None)
    if smi_col is None or gap_col is None:
        raise RuntimeError(f"need a smiles column and a gap column; saw {list(names)}")

    raw = []
    for r in rows:
        try:
            raw.append(float(r[gap_col]))
        except (TypeError, ValueError):
            raw.append(float("nan"))
    finite = sorted(v for v in raw if v == v)
    med = finite[len(finite) // 2] if finite else 0.0
    scale = HARTREE_TO_EV if med < 2.0 else 1.0
    print(f"median raw gap {med:.4f} -> unit {'Hartree' if scale > 1 else 'eV'}")

    homo_col = "homo" if "homo" in names else None
    lumo_col = "lumo" if "lumo" in names else None

    n_bad = 0
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "gap_ev", "homo_ev", "lumo_ev"])
        n = 0
        for r, g in zip(rows, raw):
            if g != g:
                n_bad += 1
                continue
            smi = (r[smi_col] or "").strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() == 0:
                n_bad += 1
                continue
            h = float(r[homo_col]) * scale if homo_col else ""
            l = float(r[lumo_col]) * scale if lumo_col else ""
            w.writerow([Chem.MolToSmiles(mol), f"{g * scale:.6f}",
                        f"{h:.6f}" if h != "" else "",
                        f"{l:.6f}" if l != "" else ""])
            n += 1
            if limit and n >= limit:
                break
    print(f"wrote {out_csv}: {n} molecules ({n_bad} skipped)")
    return out_csv
