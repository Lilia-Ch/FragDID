from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Crippen

RDLogger.DisableLog("rdApp.*")

N_DYNAMIC_FEATURES = 2          # degree and implicit-H, computed in the model
_CHARGED = Chem.MolFromSmarts(
    "[+1!H0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")


def _oneutralize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for (idx,) in m.GetSubstructMatches(_CHARGED):
        a = m.GetAtomWithIdx(idx)
        chg = a.GetFormalCharge()
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(max(0, a.GetTotalNumHs() - chg))
        a.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return Chem.MolToSmiles(m)


def _neutralize_mol(mol):
    m = Chem.Mol(mol)
    for (idx,) in m.GetSubstructMatches(_CHARGED):
        a = m.GetAtomWithIdx(idx)
        chg = a.GetFormalCharge()
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(max(0, a.GetTotalNumHs() - chg))
        a.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m

def feature_dim(elements: Sequence[str]) -> int:
    return len(elements) + 3 + N_DYNAMIC_FEATURES


def static_atom_features(atom: Chem.Atom, elements: Sequence[str]) -> List[float]:
    v = [0.0] * len(elements)
    s = atom.GetSymbol()
    if s in elements:
        v[elements.index(s)] = 1.0
    return v + [atom.GetFormalCharge() / 2.0,
                float(atom.GetIsAromatic()),
                float(atom.IsInRing())]


def mol_to_tensors(mol: Chem.Mol, elements: Sequence[str],
                   n_pad: Optional[int] = None):
    n = mol.GetNumAtoms()
    n_pad = n_pad or n
    if n > n_pad:
        return None
    x = torch.zeros(n_pad, feature_dim(elements))
    order = torch.zeros(n_pad, n_pad)
    presence = torch.zeros(n_pad, n_pad)
    mask = torch.zeros(n_pad)
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetSymbol() not in elements:
            return None
        x[i, :len(elements) + 3] = torch.tensor(static_atom_features(a, elements))
        mask[i] = 1.0
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        o = b.GetBondTypeAsDouble()
        order[i, j] = order[j, i] = o
        presence[i, j] = presence[j, i] = 1.0
    return x, order, presence, mask

def _components(mol: Chem.Mol, allowed: set,
                cut: Optional[set] = None) -> List[Tuple[int, ...]]:
    cut = cut or set()
    seen, comps = set(), []
    for start in sorted(allowed):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                v = nb.GetIdx()
                if v in allowed and v not in seen and (min(u, v), max(u, v)) not in cut:
                    seen.add(v)
                    stack.append(v)
        comps.append(tuple(sorted(comp)))
    return comps


def fragment_brics(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    cut = {(min(b[0]), max(b[0])) for b in BRICS.FindBRICSBonds(mol)}
    return _components(mol, set(range(mol.GetNumAtoms())), cut)


def fragment(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    groups = fragment_brics(mol)
    if not groups:
        return []
    # partition and connectivity invariants
    if {a for g in groups for a in g} != set(range(mol.GetNumAtoms())):
        return []
    for g in groups:
        if len(_components(mol, set(g))) != 1:
            return []
    return groups


def _submol(mol: Chem.Mol, atoms: Sequence[int]):
    atoms = sorted(atoms)
    aset = set(atoms)
    em = Chem.RWMol()
    parent_to_local: Dict[int, int] = {}
    for p in atoms:
        a = mol.GetAtomWithIdx(p)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        parent_to_local[p] = em.AddAtom(na)
    for p in atoms:
        for nb in mol.GetAtomWithIdx(p).GetNeighbors():
            q = nb.GetIdx()
            if q in aset and p < q:
                em.AddBond(parent_to_local[p], parent_to_local[q],
                           mol.GetBondBetweenAtoms(p, q).GetBondType())
    sub = em.GetMol()
    try:
        Chem.SanitizeMol(sub)
        smi = Chem.MolToSmiles(sub)
        raw = sub.GetProp("_smilesAtomOutputOrder")
        out_order = [int(v) for v in raw.strip("[]").split(",") if v.strip()]
    except Exception:
        return None
    if len(out_order) != sub.GetNumAtoms():
        return None
    local_to_canon = {loc: pos for pos, loc in enumerate(out_order)}
    return smi, {p: local_to_canon[l] for p, l in parent_to_local.items()}


def decompose(mol: Chem.Mol):
    groups = fragment(mol)
    if not groups:
        return None
    atom_to_slot = {a: k for k, g in enumerate(groups) for a in g}
    work = Chem.Mol(mol)
    try:
        Chem.Kekulize(work, clearAromaticFlags=True)
    except Exception:
        return None

    smiles, maps = [], []
    for g in groups:
        res = _submol(work, g)
        if res is None:
            return None
        smiles.append(res[0])
        maps.append(res[1])

    bonds = []
    for b in work.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        si, sj = atom_to_slot[i], atom_to_slot[j]
        if si == sj:
            continue
        if b.GetBondType() != Chem.BondType.SINGLE:
            return None         
        bonds.append((si, maps[si][i], sj, maps[sj][j]))
    return smiles, maps, bonds, groups

def build(smiles: Iterable[str], out_dir: str,
          min_count: int = 100, max_motif_atoms: int = 18,
          neutralize: bool = False, verbose: bool = True) -> "Vocab":
    counts: Counter = Counter()
    ports: Dict[str, set] = defaultdict(set)
    elements: Counter = Counter()
    n_ok = n_seen = 0

    for smi in smiles:
        n_seen += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if neutralize:
            mol = _neutralize_mol(mol)
            if mol is None:
                continue
        res = decompose(mol)
        if res is None:
            continue
        motifs, _, bonds, _ = res
        if any(Chem.MolFromSmiles(m) is None
               or Chem.MolFromSmiles(m).GetNumAtoms() > max_motif_atoms
               for m in motifs):
            continue
        n_ok += 1
        for m in set(motifs):
            counts[m] += 1
        for si, ai, sj, aj in bonds:
            ports[motifs[si]].add(ai)
            ports[motifs[sj]].add(aj)
        for a in mol.GetAtoms():
            elements[a.GetSymbol()] += 1
        if verbose and n_seen % 20000 == 0:
            print(f"  scanned {n_seen:>8}  kept {n_ok:>8}  motifs {len(counts):>6}",
                  flush=True)

    kept = sorted(m for m, c in counts.items() if c >= min_count and ports.get(m))
    if not kept:
        raise RuntimeError("empty vocabulary -- lower --min-count")
    data = {
        "elements": [e for e, _ in elements.most_common()],
        "max_motif_atoms": max_motif_atoms,
        "motifs": kept,
        "counts": [counts[m] for m in kept],
        "ports": [sorted(ports[m]) for m in kept],
        "n_molecules": n_ok,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(data, f)

    if verbose:
        cov = sum(counts[m] for m in kept) / max(sum(counts.values()), 1)
        sizes = [Chem.MolFromSmiles(m).GetNumAtoms() for m in kept]
        print(f"BRICS vocabulary: {len(kept)} motifs from {n_ok}/{n_seen} "
              f"molecules, {cov:.1%} occurrence coverage")
        print(f"  mean {np.mean(sizes):.2f} heavy atoms per motif")
    return Vocab(out_dir)

class Vocab:
    def __init__(self, path: str, device: str = "cpu"):
        fn = path if path.endswith(".json") else os.path.join(path, "vocab.json")
        with open(fn) as f:
            d = json.load(f)

        self.path = path
        self.elements: List[str] = d["elements"]
        self.motifs: List[str] = d["motifs"]
        self.counts = np.asarray(d["counts"], dtype=np.float64)
        self.index = {s: i for i, s in enumerate(self.motifs)}
        self.device = torch.device(device)

        T = len(self.motifs)
        M = max(Chem.MolFromSmiles(m).GetNumAtoms() for m in self.motifs)
        Ds = len(self.elements) + 3
        self.T, self.M, self.static_dim = T, M, Ds
        self.dim = feature_dim(self.elements)

        feat = torch.zeros(T, M, Ds)
        order = torch.zeros(T, M, M)
        pres = torch.zeros(T, M, M)
        amask = torch.zeros(T, M)
        port = torch.zeros(T, M)
        free_h = torch.zeros(T, M)
        n_atoms = torch.zeros(T)

        self.mols: List[Chem.Mol] = []
        for t, smi in enumerate(self.motifs):
            mol = Chem.MolFromSmiles(smi)
            self.mols.append(mol)
            n_atoms[t] = mol.GetNumAtoms()
            for i, a in enumerate(mol.GetAtoms()):
                feat[t, i] = torch.tensor(static_atom_features(a, self.elements))
                amask[t, i] = 1.0
                free_h[t, i] = a.GetTotalNumHs()
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                o = b.GetBondTypeAsDouble()
                order[t, i, j] = order[t, j, i] = o
                pres[t, i, j] = pres[t, j, i] = 1.0
            for p in d["ports"][t]:
                if p < M and free_h[t, p] > 0:
                    port[t, p] = 1.0

        keep = port.sum(1) > 0
        if not bool(keep.all()):
            sel = torch.nonzero(keep).flatten().tolist()
            self.motifs = [self.motifs[i] for i in sel]
            self.mols = [self.mols[i] for i in sel]
            self.counts = self.counts[sel]
            self.index = {s: i for i, s in enumerate(self.motifs)}
            feat, order, pres = feat[keep], order[keep], pres[keep]
            amask, port = amask[keep], port[keep]
            free_h, n_atoms = free_h[keep], n_atoms[keep]
            self.T = len(self.motifs)

        p = np.sqrt(self.counts)
        self.prior = torch.tensor(p / p.sum(), dtype=torch.float32)
        self.frag_logp = torch.tensor(
            [_crippen(m) for m in self.mols], dtype=torch.float32)

        self.feat_static, self.bond_order, self.bond_pres = feat, order, pres
        self.atom_mask, self.is_port, self.free_h = amask, port, free_h
        self.n_atoms = n_atoms
        self.port_list = [[p for p in range(self.M) if float(port[t, p]) > 0]
                          for t in range(self.T)]
        self.free_h_of = [{p: int(free_h[t, p]) for p in self.port_list[t]}
                          for t in range(self.T)]
        self.cap_of = [sum(self.free_h_of[t].values()) for t in range(self.T)]

        self.to(device)

    def to(self, device) -> "Vocab":
        self.device = torch.device(device)
        for n in ("feat_static", "bond_order", "bond_pres", "atom_mask","is_port", "free_h", "n_atoms", "prior", "frag_logp"):
            setattr(self, n, getattr(self, n).to(self.device))
        return self

    def ports_of(self, t: int) -> List[int]:
        return torch.nonzero(self.is_port[t]).flatten().tolist()

    def __len__(self) -> int:
        return self.T

    def __repr__(self) -> str:
        return (f"Vocab(BRICS, {self.T} motifs, max {self.M} atoms/motif, "
                f"{len(self.elements)} elements, "
                f"mean {self.n_atoms.mean():.1f} atoms/motif)")


def _crippen(mol) -> float:
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return 0.0
