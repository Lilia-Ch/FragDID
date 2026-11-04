from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from .vocab import Vocab, mol_to_tensors
from .metrics import BRENK, PAINS, HAVE_SA

RDLogger.DisableLog("rdApp.*")
if HAVE_SA:
    import sascorer                                         


@dataclass
class DesignConfig:
    target: Tuple[float, float] = (5.0, 5.5)
    min_atoms: int = 12
    max_atoms: int = 38
    n_slots: int = 12
    batch: int = 48
    steps: int = 300
    max_rounds: int = 40
    lr: float = 0.05
    edge_threshold: float = 0.5
    resume_frac: float = 0.5      
    archive_every: int = 5     
    prior_beta: float = 1.0       
    init_noise: float = 1.0
    seed: Optional[int] = 0
    w_prop: float = 1.0           
    w_size: float = 0.05          
    prop_scale: float = 1.0       
    max_sa: float = 4.5
    max_ring: int = 7
    allow_pains: bool = False
    allow_brenk: bool = False
    max_per_scaffold: int = 5
    neutral_only: bool = False
    oracle: Optional[object] = None
    rank_by: str = "pred"        
    trace: int = 0              
    novel_only: bool = True
    known: Optional[frozenset] = None     
    use_ste: bool = True          
    restore_best: bool = True     
    flat_edge_init: bool = False  
    tree_decode: bool = True      
class Candidates:
    def __init__(self, vocab: Vocab, cfg: DesignConfig,
                 affinity: Optional[torch.Tensor] = None):
        self._affinity = affinity
        B, N, T, M = cfg.batch, cfg.n_slots, vocab.T, vocab.M
        dev = vocab.device
        log_prior = torch.log(vocab.prior.clamp_min(1e-12))
        bias = cfg.prior_beta * log_prior.view(1, 1, T)
        if affinity is not None:      # target-aware initialisation
            bias = bias + affinity.view(1, 1, T)
        self.S = (bias + cfg.init_noise * torch.randn(B, N, T, device=dev)
                  ).requires_grad_(True)
        with torch.no_grad():
            p_init = torch.softmax(bias[0, 0], dim=-1)
            exp_atoms = float((p_init * vocab.n_atoms).sum())
        want = 0.5 * (cfg.min_atoms + cfg.max_atoms)
        k_mid = int(round(want / max(exp_atoms, 1.0)))
        k_lo = max(2, min(N, k_mid - 1))
        k_hi = max(k_lo + 1, min(N, k_mid + 2))
        A0 = torch.full((B, N, N), -2.0, device=dev)   
        for b in range(B):
            k = int(torch.randint(k_lo, k_hi + 1, (1,)).item())
            if cfg.flat_edge_init:
                for _ in range(max(k - 1, 0)):
                    i, j = torch.randperm(N, device=dev)[:2].tolist()
                    A0[b, i, j] = A0[b, j, i] = 1.0
                continue
            nodes = torch.randperm(N, device=dev)[:k].tolist()
            for ii in range(1, k):
                child = nodes[ii]
                parent = nodes[int(torch.randint(0, ii, (1,)).item())]
                A0[b, parent, child] = 1.0               
                A0[b, child, parent] = 1.0
        A0 = A0 + 0.2 * torch.randn_like(A0)
        self.A = torch.triu(A0, 1).clone().requires_grad_(True)
        self.P = torch.zeros(B, N, N, M, device=dev)
        self.vocab, self.cfg = vocab, cfg

    def params(self):
        return [self.S, self.A]

    def soft(self):
        S = torch.softmax(self.S, dim=-1)
        au = torch.triu(self.A, 1)
        A = torch.sigmoid(au + au.transpose(1, 2))
        A = A * (1 - torch.eye(A.size(1), device=A.device)).unsqueeze(0)
        port_mask = torch.einsum("bnt,tm->bnm", S, self.vocab.is_port)
        P = torch.softmax(self.P.masked_fill(
            (port_mask < 1e-6).unsqueeze(2), -1e9), dim=-1)
        return S, A, P

    def reseed(self, keep: torch.Tensor, S, A) -> None:
        fresh = Candidates(self.vocab, self.cfg, self._affinity)
        with torch.no_grad():
            k = keep.view(-1, 1, 1)
            self.S.copy_(torch.where(k, S + 0.3 * torch.randn_like(S), fresh.S))
            self.A.copy_(torch.where(k, A + 0.3 * torch.randn_like(A), fresh.A))

@dataclass
class Decoded:
    smiles: Optional[str]
    types: List[int]
    active: List[int]
    edges: List[Tuple[int, int, int, int]] = field(default_factory=list)

def _pick_port(budget: Dict[int, int]) -> Optional[int]:
    free = [p for p, b in budget.items() if b > 0]
    if not free:
        return None
    return free[random.randrange(len(free))]


def decode_one(vocab: Vocab, cfg: DesignConfig, S, A) -> Decoded:
    N = S.size(0)
    types = S.argmax(-1).tolist()
    Av = A.detach().cpu().tolist()
    keep = [i for i in range(N)
            if max(Av[i]) > cfg.edge_threshold] or [0]
    cap = {i: vocab.cap_of[types[i]] for i in keep}

    if cfg.tree_decode:
        tree, seen = [], {keep[0]}
        while len(seen) < len(keep):
            best = None
            for i in seen:
                if cap[i] <= 0:
                    continue
                for j in keep:
                    if j in seen or cap[j] <= 0:
                        continue
                    w = Av[i][j]
                    if w > cfg.edge_threshold and (best is None or w > best[0]):
                        best = (w, i, j)
            if best is None:
                break
            _, i, j = best
            tree.append((i, j))
            seen.add(j)
            cap[i] -= 1
            cap[j] -= 1
        active = sorted(seen)
    else:
        cand = sorted(((Av[i][j], i, j)
                       for xi, i in enumerate(keep) for j in keep[xi + 1:]
                       if Av[i][j] > cfg.edge_threshold),
                      key=lambda e: -e[0])
        tree = []
        for _, i, j in cand:
            if cap[i] > 0 and cap[j] > 0:
                tree.append((i, j))
                cap[i] -= 1
                cap[j] -= 1
        adj: Dict[int, List[int]] = {i: [] for i in keep}
        for i, j in tree:
            adj[i].append(j)
            adj[j].append(i)
        comp_best: List[int] = []
        visited = set()
        for s in keep:
            if s in visited:
                continue
            comp, stack = [], [s]
            visited.add(s)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            if len(comp) > len(comp_best):
                comp_best = comp
        active = sorted(comp_best or [keep[0]])
        aset = set(active)
        tree = [(i, j) for i, j in tree if i in aset and j in aset]

    rw = Chem.RWMol()
    offset, budget = {}, {}
    for i in active:
        t = types[i]
        m = vocab.mols[t]
        base = rw.GetNumAtoms()
        offset[i] = base
        for a in m.GetAtoms():
            na = Chem.Atom(a.GetAtomicNum())
            na.SetFormalCharge(a.GetFormalCharge())
            na.SetNoImplicit(False)
            rw.AddAtom(na)
        for b in m.GetBonds():
            rw.AddBond(base + b.GetBeginAtomIdx(), base + b.GetEndAtomIdx(),
                       b.GetBondType())
        budget[i] = dict(vocab.free_h_of[t])

    edges = []
    for i, j in tree:
        pi = _pick_port(budget[i])
        pj = _pick_port(budget[j])
        if pi is None or pj is None:
            continue
        budget[i][pi] -= 1
        budget[j][pj] -= 1
        rw.AddBond(offset[i] + pi, offset[j] + pj, Chem.BondType.SINGLE)
        edges.append((i, pi, j, pj))

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        if "." in smi or Chem.MolFromSmiles(smi) is None:
            smi = None
    except Exception:
        smi = None
    return Decoded(smi, types, active, edges)


def hard_tensors(vocab: Vocab, cfg: DesignConfig, decs: List[Decoded], device):
    B, N, T, M = len(decs), cfg.n_slots, vocab.T, vocab.M
    S = torch.zeros(B, N, T)
    A = torch.zeros(B, N, N)
    P = torch.zeros(B, N, N, M)
    act = torch.zeros(B, N)
    for b, d in enumerate(decs):
        for i, t in enumerate(d.types):
            S[b, i, t] = 1.0
        for i in d.active:
            act[b, i] = 1.0
        for i, pi, j, pj in d.edges:
            A[b, i, j] = A[b, j, i] = 1.0
            P[b, i, j, pi] = 1.0
            P[b, j, i, pj] = 1.0
    return S.to(device), A.to(device), P.to(device), act.to(device)


def straight_through(soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    return hard + soft - soft.detach()


def assemble(vocab: Vocab, S, A, P, act):
    B, N, T = S.shape
    M = vocab.M
    dev = S.device
    Sg = S * act.unsqueeze(-1)

    x = torch.einsum("bnt,tmd->bnmd", Sg, vocab.feat_static)
    mask = torch.einsum("bnt,tm->bnm", Sg, vocab.atom_mask)
    intra_o = torch.einsum("bnt,tmk->bnmk", Sg, vocab.bond_order)
    intra_p = torch.einsum("bnt,tmk->bnmk", Sg, vocab.bond_pres)
    free_h = torch.einsum("bnt,tm->bnm", Sg, vocab.free_h)

    Ag = A * act.unsqueeze(2) * act.unsqueeze(1)
    E = (Ag.unsqueeze(-1).unsqueeze(-1) * P.unsqueeze(-1)
         * P.permute(0, 2, 1, 3).unsqueeze(-2))

    order = torch.zeros(B, N, M, N, M, device=dev)
    pres = torch.zeros(B, N, M, N, M, device=dev)
    idx = torch.arange(N, device=dev)
    order[:, idx, :, idx, :] = intra_o.permute(1, 0, 2, 3)
    pres[:, idx, :, idx, :] = intra_p.permute(1, 0, 2, 3)
    inter = E.permute(0, 1, 3, 2, 4)
    order = order + inter
    pres = pres + inter

    n = N * M
    x = x.reshape(B, n, -1)
    mask = mask.reshape(B, n)
    order = order.reshape(B, n, n) * mask.unsqueeze(1) * mask.unsqueeze(2)
    pres = pres.reshape(B, n, n) * mask.unsqueeze(1) * mask.unsqueeze(2)
    ext_deg = (Ag.unsqueeze(-1) * P).sum(dim=2)         
    aux = {"ext_deg": ext_deg, "free_h": free_h,
           "atom_mask": mask.reshape(B, N, M), "n_atoms": mask.sum(dim=1)}
    return x, order, pres, mask, aux



def objective(cfg: DesignConfig, pred, aux):
    lo, hi = cfg.target
    centre = 0.5 * (lo + hi)
    scale = float(getattr(cfg, "prop_scale", 1.0)) or 1.0
    L_prop = ((pred - centre) / scale) ** 2

    n = aux["n_atoms"]
    L_size = F.relu(cfg.min_atoms - n) ** 2 + F.relu(n - cfg.max_atoms) ** 2

    total = cfg.w_prop * L_prop + cfg.w_size * L_size
    return total, {"prop": L_prop, "size": L_size, "n_atoms": n}

class Archive:
    def __init__(self, cfg: DesignConfig):
        self.cfg = cfg
        self.items: Dict[str, dict] = {}
        self.rejected: Dict[str, int] = {}
        self.known = frozenset(getattr(cfg, "known", None) or ())

    def _pool(self):
        if not (getattr(self.cfg, "novel_only", True) and self.known):
            return list(self.items.values())
        return [r for r in self.items.values() if r["smiles"] not in self.known]

    def _reject(self, why: str) -> None:
        self.rejected[why] = self.rejected.get(why, 0) + 1

    def add(self, smiles: Optional[str], pred: float, std: float = 0.0) -> None:
        if not smiles or smiles in self.items:
            return
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            self._reject("invalid")
            return
        n_heavy = mol.GetNumHeavyAtoms()
        if not (self.cfg.min_atoms <= n_heavy <= self.cfg.max_atoms):
            self._reject("size")
            return
        if max((len(r) for r in mol.GetRingInfo().AtomRings()),
               default=0) > self.cfg.max_ring:
            self._reject("ring")
            return
        if getattr(self.cfg, "neutral_only", False) and \
                Chem.GetFormalCharge(mol) != 0:
            self._reject("charged")
            return
        self.items[smiles] = {"smiles": smiles, "pred": float(pred),
                              "std": float(std), "n_atoms": n_heavy,
                              "_full": False}

    def _enrich(self, rec: dict) -> dict:
        if rec["_full"]:
            return rec
        mol = Chem.MolFromSmiles(rec["smiles"])
        rec["sa"] = sascorer.calculateScore(mol) if HAVE_SA else 0.0
        rec["qed"] = QED.qed(mol)
        rec["pains"] = int(PAINS.HasMatch(mol))
        rec["brenk"] = int(BRENK.HasMatch(mol))
        try:
            rec["scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            rec["scaffold"] = ""
        if self.cfg.oracle is not None:
            rec["true"] = float(self.cfg.oracle(mol))
        rec["_full"] = True
        return rec

    def _miss(self, r: dict) -> float:
        lo, hi = self.cfg.target
        v = r["true"] if (self.cfg.rank_by == "oracle" and "true" in r) else r["pred"]
        return max(0.0, lo - v, v - hi)

    def n_hits(self) -> int:
        return sum(1 for r in self._pool() if self._miss(r) <= 0)

    def best(self, n: int) -> List[dict]:
        """Rank on property miss, then apply quality filters, then cap
        scaffolds. Enrichment is restricted to the shortlist."""
        pool = self._pool()
        if getattr(self.cfg, "novel_only", True) and self.known:
            self.rejected["known"] = len(self.items) - len(pool)
        shortlist = sorted(pool,
                           key=lambda r: (self._miss(r), r["std"]))[:max(20 * n, 400)]
        keep = []
        for r in (self._enrich(x) for x in shortlist):
            if r["sa"] > self.cfg.max_sa:
                self._reject("sa"); continue
            if not self.cfg.allow_pains and r["pains"]:
                self._reject("pains"); continue
            if not self.cfg.allow_brenk and r["brenk"]:
                self._reject("brenk"); continue
            keep.append(r)
        keep.sort(key=lambda r: (self._miss(r), r["std"], r["sa"], -r["qed"]))
        out, seen = [], {}
        for r in keep:
            s = r["scaffold"]
            if seen.get(s, 0) >= self.cfg.max_per_scaffold:
                continue
            seen[s] = seen.get(s, 0) + 1
            out.append(r)
            if len(out) >= n:
                break
        return out

    def all_records(self) -> List[dict]:
        return [self._enrich(r) for r in self.items.values()]

    def __len__(self):
        return len(self.items)


TRACE_HEADER = (f"{'step':>5} {'loss':>8} {'pred':>8} {'win%':>6} {'hit%':>6} "
                f"{'atoms':>6} {'valid':>6} {'uniq':>6} {'SA':>5} {'found':>7}")


class Designer:
    def __init__(self, model, vocab: Vocab, cfg: DesignConfig, device="cpu"):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vocab = vocab.to(device)
        self.cfg = cfg
        self.device = device
        self.affinity = None
        self.n_decodes = 0         

    @torch.no_grad()
    def _motif_property(self) -> torch.Tensor:
        M = self.vocab.M
        n = len(self.vocab.mols)
        x = torch.zeros(n, M, self.vocab.dim, device=self.device)
        mask = torch.zeros(n, M, device=self.device)
        x[:, :, :self.vocab.static_dim] = self.vocab.feat_static
        mask[:] = self.vocab.atom_mask
        return self.model(x, self.vocab.bond_order, self.vocab.bond_pres,
                          mask).view(-1).detach()

    def set_affinity(self, mode: str = "model", gamma: float = 1.0) -> None:
        self.affinity = None          
        if mode == "none" or gamma <= 0.0:
            return
        if mode != "model":
            raise SystemExit(f"affinity mode '{mode}' is not supported; "
                             f"the bias always comes from the predictor itself")

        lo, hi = self.cfg.target
        centre = 0.5 * (lo + hi)
        contrib = self._motif_property()

        if self.model.readout == "sum":
            p_init = torch.softmax(
                self.cfg.prior_beta
                * torch.log(self.vocab.prior.clamp_min(1e-12)), dim=-1)
            mean_atoms = float((p_init * self.vocab.n_atoms).sum())
            want = 0.5 * (self.cfg.min_atoms + self.cfg.max_atoms)
            centre = centre / max(2.0, round(want / max(mean_atoms, 1.0)))

        d = (contrib - centre) ** 2
        self.affinity = -gamma * d / d.std().clamp(min=1e-6)

    def forward(self, cand: Candidates, hard_only=False):
        S_s, A_s, P_s = cand.soft()
        S_d, A_d = S_s.detach().cpu(), A_s.detach().cpu()
        decs = [decode_one(self.vocab, self.cfg, S_d[b], A_d[b])
                for b in range(S_s.size(0))]
        self.n_decodes += len(decs)
        S_h, A_h, P_h, act_h = hard_tensors(self.vocab, self.cfg, decs, self.device)
        act_s = 1.0 - torch.prod(1.0 - A_s.clamp(0, 1), dim=2)
        if hard_only:
            S, A, P, act = S_h, A_h, P_h, act_h
        elif self.cfg.use_ste:
            S = straight_through(S_s, S_h)
            A = straight_through(A_s, A_h)
            P = P_h
            act = straight_through(act_s, act_h)
        else:
            S, A, P, act = S_s, A_s, P_s, act_s
        x, order, pres, mask, aux = assemble(self.vocab, S, A, P, act)
        pred, std = self.model(x, order, pres, mask, return_std=True)
        return pred, std, decs, aux

    def check_consistency(self, n=8) -> float:
        cand = Candidates(self.vocab, self.cfg)
        with torch.no_grad():
            pred, _, decs, _ = self.forward(cand, hard_only=True)
        worst, checked = 0.0, 0
        for b, d in enumerate(decs):
            if not d.smiles or checked >= n:
                continue
            mol = Chem.MolFromSmiles(d.smiles)
            t = mol_to_tensors(mol, self.vocab.elements,
                               self.cfg.n_slots * self.vocab.M)
            if t is None:
                continue
            x, o, p, m = [z.unsqueeze(0).to(self.device) for z in t]
            with torch.no_grad():
                ref = self.model(x, o, p, m).view(-1)[0]
            worst = max(worst, abs(float(pred[b] - ref)))
            checked += 1
        print(f"assemble/decode consistency: {checked} molecules, "
              f"max |dpred| = {worst:.2e}")
        return worst

    def check_valence(self, n_batches: int = 5) -> dict:
        worst, total, n = 0.0, 0.0, 0
        for _ in range(n_batches):
            cand = Candidates(self.vocab, self.cfg, self.affinity)
            with torch.no_grad():
                _, _, decs, aux = self.forward(cand, hard_only=True)
                over = F.relu(aux["ext_deg"] - aux["free_h"]) * aux["atom_mask"]
                worst = max(worst, float(over.max()))
                total += float(over.sum())
                n += len(decs)
        print(f"valence: {n} decodes, total violation {total:.3g}, "
              f"max {worst:.3g}")
        return {"decodes": n, "total_violation": total, "max_violation": worst}

    def random_baseline(self, n: int = 300) -> dict:
        cfg = self.cfg
        lo, hi = cfg.target
        vals, n_valid, n_try = [], 0, 0
        while n_try < n:
            cand = Candidates(self.vocab, cfg, self.affinity)
            S, A, _ = cand.soft()
            S, A = S.detach().cpu(), A.detach().cpu()
            for b in range(S.size(0)):
                n_try += 1
                d = decode_one(self.vocab, cfg, S[b], A[b])
                if not d.smiles:
                    continue
                n_valid += 1
                if cfg.oracle is not None:
                    m = Chem.MolFromSmiles(d.smiles)
                    if m is not None:
                        vals.append(float(cfg.oracle(m)))
        hit = sum(1 for v in vals if lo <= v <= hi) / max(len(vals), 1)
        return {"decoded": n_valid, "tried": n_try,
                "decode_rate": n_valid / max(n_try, 1),
                "oracle_mean": sum(vals) / len(vals) if vals else float("nan"),
                "random_success": hit}

    def _batch_report(self, decs, pred) -> dict:
        lo, hi = self.cfg.target
        smis = [d.smiles for d in decs if d.smiles]
        rep = {"valid": len(smis) / max(len(decs), 1),
               "uniq": len(set(smis)) / max(len(smis), 1) if smis else 0.0,
               "win": float(((pred >= lo) & (pred <= hi)).float().mean()),
               "hit": float("nan"), "sa": float("nan")}
        mols = [m for m in (Chem.MolFromSmiles(s) for s in smis[:24])
                if m is not None]
        if HAVE_SA and mols:
            rep["sa"] = sum(sascorer.calculateScore(m) for m in mols) / len(mols)
        if self.cfg.oracle is not None and mols:
            vals = [float(self.cfg.oracle(m)) for m in mols]
            rep["hit"] = sum(lo <= v <= hi for v in vals) / len(vals)
        return rep

    def run(self, n_molecules: int, verbose=True) -> Archive:
        cfg = self.cfg
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            random.seed(cfg.seed)
        arch = Archive(cfg)
        self.n_decodes = 0
        t0 = time.time()
        cand, best_score = None, None

        for r in range(cfg.max_rounds):
            if arch.n_hits() >= n_molecules:
                break

            if cand is None or cfg.resume_frac <= 0:
                cand = Candidates(self.vocab, cfg, self.affinity)
            else:
                order = torch.argsort(best_score)
                keep = torch.zeros_like(best_score, dtype=torch.bool)
                keep[order[:int(cfg.resume_frac * cfg.batch)]] = True
                cand.reseed(keep, best_S, best_A)

            opt = torch.optim.Adam(cand.params(), lr=cfg.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(cfg.steps, 1), eta_min=cfg.lr * 0.05)
            B = cfg.batch
            best_score = torch.full((B,), float("inf"), device=self.device)
            best_S = cand.S.detach().clone()
            best_A = cand.A.detach().clone()
            best_step = torch.full((B,), -1, dtype=torch.long, device=self.device)

            if cfg.trace:
                print(f"\n--- round {r + 1} ---")
                print(TRACE_HEADER)

            for step in range(cfg.steps):
                pred, std, decs, aux = self.forward(cand)
                loss, parts = objective(cfg, pred, aux)
                score = loss.detach()
                improved = score < best_score
                if improved.any():
                    idx = torch.where(improved)[0]
                    best_score[idx] = score[idx]
                    best_S[idx] = cand.S.detach()[idx]
                    best_A[idx] = cand.A.detach()[idx]
                    best_step[idx] = step

                opt.zero_grad(set_to_none=True)
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(cand.params(), 2.0)
                opt.step()
                sched.step()
                if step % max(cfg.archive_every, 1) == 0:
                    pv = pred.detach().cpu().tolist()
                    sv = std.detach().cpu().tolist()
                    for b, d in enumerate(decs):
                        arch.add(d.smiles, pv[b], sv[b])

                if cfg.trace and (step % cfg.trace == 0 or step == cfg.steps - 1):
                    rep = self._batch_report(decs, pred.detach())
                    hit = "     ." if rep["hit"] != rep["hit"] else f"{rep['hit']:>6.2f}"
                    sa = "    ." if rep["sa"] != rep["sa"] else f"{rep['sa']:>5.2f}"
                    print(f"{step:>5} {float(loss.mean()):>8.3f} "
                          f"{float(pred.mean()):>8.2f} {rep['win']:>6.2f} {hit} "
                          f"{float(parts['n_atoms'].mean()):>6.1f} "
                          f"{rep['valid']:>6.2f} {rep['uniq']:>6.2f} {sa} "
                          f"{len(arch):>7}", flush=True)

            if cfg.restore_best and cfg.steps > 0:
                with torch.no_grad():
                    cand.S.copy_(best_S)
                    cand.A.copy_(best_A)
            with torch.no_grad():
                pred, std, decs, _ = self.forward(cand)
            pv = pred.detach().cpu().tolist()
            sv = std.detach().cpu().tolist()
            for b, d in enumerate(decs):
                arch.add(d.smiles, pv[b], sv[b])

            if cfg.trace:
                vs = best_step[best_step >= 0]
                ms = float(vs.float().mean()) if len(vs) else float("nan")
                rep = self._batch_report(decs, pred)
                print(f"  restored best: mean peak at step {ms:.0f}/{cfg.steps}"
                      f"   win {rep['win']:.2f}   valid {rep['valid']:.2f}")
                if ms == ms and ms < 0.3 * cfg.steps:
                    print("  candidates peak early: the schedule anneals past "
                          "the optimum, try fewer steps or a lower lr")
                elif ms == ms and ms > 0.85 * cfg.steps:
                    print("  candidates peak at the end: not converged, "
                          "try more steps")

            if verbose:
                print(f"  round {r + 1}: {len(arch)} found, "
                      f"{arch.n_hits()} in window", flush=True)

        self.elapsed = time.time() - t0
        return arch
