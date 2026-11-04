from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED

from .design import Designer, DesignConfig
from .metrics import canonical_set, evaluate
from .model import PropertyGNN
from .vocab import Vocab, mol_to_tensors

RDLogger.DisableLog("rdApp.*")

ORACLES = {"logp": Crippen.MolLogP, "qed": QED.qed, "none": None}


def load_stack(vocab_path: str, model_path: str, device: str = "cpu"):
    return Vocab(vocab_path, device=device), PropertyGNN.load(model_path, device=device)


@torch.no_grad()
def score_smiles(model: PropertyGNN, smiles: Sequence[str], device: str = "cpu",
                 batch: int = 128) -> List[float]:
    mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
    out = [float("nan")] * len(smiles)
    idx = [i for i, m in enumerate(mols) if m is not None]
    if not idx:
        return out
    n_pad = max(mols[i].GetNumAtoms() for i in idx)
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        ts, keep = [], []
        for i in chunk:
            t = mol_to_tensors(mols[i], model.elements, n_pad)
            if t is not None:
                ts.append(t)
                keep.append(i)
        if not ts:
            continue
        x, o, p, m = [torch.stack([t[k] for t in ts]).to(device) for k in range(4)]
        mu = model(x, o, p, m).view(-1).tolist()
        for i, v in zip(keep, mu):
            out[i] = float(v)
    return out


def generate(cfg: DesignConfig, vocab: Vocab, model: PropertyGNN, n: int,
             device: str = "cpu", affinity: str = "model",
             affinity_gamma: float = 1.0,
             eval_model: Optional[PropertyGNN] = None,
             reference: Optional[Sequence[str]] = None,
             oracle_name: str = "logp", out_dir: Optional[str] = None,
             check: bool = False, baseline: bool = False,
             n_stop: Optional[int] = None, full_metrics: bool = False,
             verbose: bool = True) -> Dict[str, object]:
    cfg.prop_scale = float(model.y_std)
    if not reference:
        raise SystemExit("a reference set is required: novel-only ranking is "
                         "always on, pass --reference")
    cfg.known = canonical_set(reference)
    if verbose:
        print(f"novel-only: {len(cfg.known)} reference molecules refused "
              f"at ranking time")
    d = Designer(model, vocab, cfg, device=device)
    d.set_affinity(affinity, gamma=affinity_gamma)
    result: Dict[str, object] = {}
    if check:
        result["consistency"] = d.check_consistency()
        result["valence"] = d.check_valence()
    if baseline:
        bl = d.random_baseline()
        result["baseline"] = bl
        if verbose:
            print(f"random control (NO optimization): decode rate "
                  f"{bl['decode_rate']:.2f}, oracle mean {bl['oracle_mean']:.2f}, "
                  f"success {bl['random_success']:.3f}")
            print("  -> optimized success must clearly beat this, or the "
                  "gradient is doing nothing")

    arch = d.run(n if n_stop is None else n_stop, verbose=verbose)
    rows = arch.best(n)
    smis = [r["smiles"] for r in rows]
    preds = [r["pred"] for r in rows]

    extern = None
    if eval_model is not None and smis:
        extern = score_smiles(eval_model, smis, device=device)
        for r, v in zip(rows, extern):
            r["eval"] = v

    stats = evaluate(smis, reference, cfg.target, preds,
                     oracle_name if oracle_name in ("logp", "qed") else "none",
                     extern, full=full_metrics) if smis else {}
    stats["n_visited"] = len(arch)
    stats["n_decoder_calls"] = d.n_decodes
    stats["seconds"] = round(getattr(d, "elapsed", 0.0), 1)
    n_hit = int(stats.get("n_in_window", stats.get("n_in_window_eval", 0)))
    if not n_hit and stats.get("success_proxy") is not None:
        n_hit = int(round(stats["success_proxy"] * len(smis)))
    stats["seconds_per_hit"] = (round(d.elapsed / max(n_hit, 1), 2)
                                if getattr(d, "elapsed", 0.0) else None)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(os.path.join(out_dir, "molecules.csv"), rows)
        _write_csv(os.path.join(out_dir, "visited.csv"), arch.all_records())
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(stats, f, indent=2)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            blob = {k: v for k, v in vars(cfg).items()
                    if isinstance(v, (int, float, str, bool, tuple, type(None)))}
            blob["affinity"] = affinity
            blob["affinity_gamma"] = affinity_gamma
            json.dump(blob, f, indent=2)

    result.update({"rows": rows, "stats": stats, "archive": arch})
    return result


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
