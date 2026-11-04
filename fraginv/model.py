from __future__ import annotations

import inspect
import json
import math
import os
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

BOND_CENTERS = (1.0, 1.5, 2.0, 3.0)
BOND_WIDTH = 0.20
DEFAULT_VALENCE = {"C": 4, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1, "Br": 1, "I": 1, "P": 3, "B": 3, "Si": 4, "Se": 2}


def bond_rbf(order: torch.Tensor) -> torch.Tensor:
    c = torch.tensor(BOND_CENTERS, dtype=order.dtype, device=order.device)
    return torch.exp(-((order.unsqueeze(-1) - c) ** 2) / (2 * BOND_WIDTH ** 2))


def mlp(sizes: Sequence[int], act=nn.SiLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)

class BondGIN(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge = nn.Linear(len(BOND_CENTERS), hidden)
        self.eps = nn.Parameter(torch.zeros(1))
        self.nn = mlp([hidden, 2 * hidden, hidden])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, rbf, pres, mask):
        e = self.edge(rbf)                                   
        msg = F.relu(h.unsqueeze(1) + e)                    
        agg = (pres.unsqueeze(-1) * msg).sum(dim=2)         
        out = self.nn((1.0 + self.eps) * h + agg)
        return self.norm(h + out) * mask.unsqueeze(-1)


class BondGCN(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.chan = nn.ModuleList(nn.Linear(hidden, hidden, bias=False)
                                  for _ in BOND_CENTERS)
        self.root = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, rbf, pres, mask):
        deg = pres.sum(-1).clamp(min=1.0)
        dinv = deg.pow(-0.5)
        agg = 0.0
        for b, lin in enumerate(self.chan):
            a = pres * rbf[..., b] * dinv.unsqueeze(2) * dinv.unsqueeze(1)
            agg = agg + lin(torch.bmm(a, h))
        return self.norm(h + F.silu(self.root(h) + agg)) * mask.unsqueeze(-1)


class BondGAT(nn.Module):
    def __init__(self, hidden: int, heads: int = 4):
        super().__init__()
        assert hidden % heads == 0
        self.h, self.d = heads, hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.edge = nn.Linear(len(BOND_CENTERS), heads)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, rbf, pres, mask):
        B, n, _ = h.shape
        q = self.q(h).view(B, n, self.h, self.d)
        k = self.k(h).view(B, n, self.h, self.d)
        v = self.v(h).view(B, n, self.h, self.d)
        logit = torch.einsum("bihd,bjhd->bijh", q, k) / (self.d ** 0.5)
        logit = logit + self.edge(rbf)
        logit = logit.masked_fill((pres <= 0).unsqueeze(-1), -1e9)
        att = torch.softmax(logit, dim=2) * (pres > 0).unsqueeze(-1)
        agg = torch.einsum("bijh,bjhd->bihd", att, v).reshape(B, n, -1)
        return self.norm(h + F.silu(self.out(agg))) * mask.unsqueeze(-1)


CONVS = {"gin": BondGIN, "gcn": BondGCN, "gat": BondGAT}


class PropertyGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, layers: int = 4,
                 readout: str = "sum", elements: Optional[List[str]] = None,
                 n_heads: int = 5, conv: str = "gin",
                 mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, hidden=hidden, layers=layers,
                        readout=readout, elements=elements or [],
                        n_heads=n_heads, conv=conv)
        Conv = CONVS[conv]
        self.embed = mlp([in_dim, hidden, hidden])
        self.layers = nn.ModuleList(Conv(hidden) for _ in range(layers))
        self.conv_name = conv
        self.heads = nn.ModuleList(mlp([hidden, hidden, 1]) for _ in range(n_heads))
        self.readout = readout
        self.elements = elements or []
        val = [DEFAULT_VALENCE.get(e, 4) for e in self.elements]
        self.register_buffer("valence", torch.tensor(val, dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(float(mean)))
        self.register_buffer("y_std", torch.tensor(float(std)))

    def _augment(self, x, order, pres, mask):
        ne = len(self.elements)
        deg = (order * pres).sum(-1)
        maxval = (x[..., :ne] * self.valence.view(1, 1, -1)).sum(-1)
        n_h = (maxval - deg).clamp(min=0.0)
        return torch.cat([x[..., :ne + 3], (deg / 4.0).unsqueeze(-1),
                          (n_h / 4.0).unsqueeze(-1)], -1) * mask.unsqueeze(-1)

    def encode(self, x, order, pres, mask):
        h = self.embed(self._augment(x, order, pres, mask)) * mask.unsqueeze(-1)
        rbf = bond_rbf(order)
        for layer in self.layers:
            h = layer(h, rbf, pres, mask)
        return h

    def _head_out(self, h, mask, head):
        pa = head(h).squeeze(-1) * mask
        s = pa.sum(dim=1)
        return s if self.readout == "sum" else s / mask.sum(dim=1).clamp(min=1.0)

    def forward(self, x, order, pres, mask, return_std=False):
        h = self.encode(x, order, pres, mask)
        stack = torch.stack([self._head_out(h, mask, hd) for hd in self.heads], 0)
        stack = stack * self.y_std + self.y_mean
        mu = stack.mean(0)
        return (mu, stack.std(0)) if return_std else mu

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"cfg": self.cfg, "state": self.state_dict(),
                    "y_mean": float(self.y_mean), "y_std": float(self.y_std)}, path)

    @staticmethod
    def load(path: str, device: str = "cpu") -> "PropertyGNN":
        blob = torch.load(path, map_location=device, weights_only=False)
        ok = set(inspect.signature(PropertyGNN.__init__).parameters)
        cfg = {k: v for k, v in dict(blob["cfg"]).items() if k in ok}
        m = PropertyGNN(**cfg, mean=blob.get("y_mean", 0.0),
                        std=blob.get("y_std", 1.0))
        m.load_state_dict(blob["state"], strict=False)
        return m.to(device).eval()


class DenseMolDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, elements, n_pad, verbose=True):
        from .vocab import mol_to_tensors
        self.items = []
        self.skipped = 0
        for mol, y in pairs:
            t = mol_to_tensors(mol, elements, n_pad)
            if t is None:
                self.skipped += 1
                continue
            self.items.append((*t, torch.tensor(float(y))))
        if verbose and self.skipped:
            print(f"  skipped {self.skipped} molecules (unfeaturisable)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _evaluate(model, loader, device):
    model.eval()
    err = sq = cnt = spread = 0.0
    ys: List[float] = []
    with torch.no_grad():
        for x, o, p, m, y in loader:
            x, o, p, m, y = [t.to(device) for t in (x, o, p, m, y)]
            mu, sd = model(x, o, p, m, return_std=True)
            d = mu.view(-1) - y.view(-1)
            err += float(d.abs().sum())
            sq += float((d ** 2).sum())
            spread += float(sd.sum())
            cnt += y.numel()
            ys.extend(y.view(-1).tolist())
    if cnt == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "std": float("nan"), "n": 0}
    var = sum((v - sum(ys) / len(ys)) ** 2 for v in ys) / len(ys)
    return {"mae": err / cnt, "rmse": math.sqrt(sq / cnt),
            "r2": 1.0 - (sq / cnt) / max(var, 1e-12),
            "std": spread / cnt, "n": int(cnt)}


def train_predictor(model, ds, epochs=30, batch=64, lr=1e-3,
                    val_frac=0.1, test_frac=0.1, device="cpu", out=None,
                    seed=0, patience=0, verbose=True):
    g = torch.Generator().manual_seed(seed)
    n_val = max(1, int(len(ds) * val_frac))
    n_test = max(1, int(len(ds) * test_frac))
    n_tr = len(ds) - n_val - n_test
    tr, va, te = torch.utils.data.random_split(ds, [n_tr, n_val, n_test], generator=g)
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch)
    dl_te = torch.utils.data.DataLoader(te, batch_size=batch)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * len(dl_tr)))
    best, bad, K = float("inf"), 0, len(model.heads)
    history = []
    if verbose:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  conv={model.conv_name}  readout={model.readout}  heads={K}  "
              f"params={n_par:,}")
        print(f"  train {len(tr)} / val {len(va)} / test {len(te)}")

    for ep in range(epochs):
        model.train()
        tot = n = 0
        for x, o, p, m, y in dl_tr:
            x, o, p, m, y = [t.to(device) for t in (x, o, p, m, y)]
            h = model.encode(x, o, p, m)
            loss = 0.0
            for head in model.heads:
                yh = model._head_out(h, m, head) * model.y_std + model.y_mean
                w = (torch.rand(y.size(0), device=device) < 0.8).float()
                loss = loss + (F.smooth_l1_loss(yh, y.view(-1), beta=0.1,
                                                reduction="none") * w
                               ).sum() / w.sum().clamp(min=1)
            loss = loss / K
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss.detach()) * y.size(0)
            n += y.size(0)

        v = _evaluate(model, dl_va, device)
        history.append({"epoch": ep + 1, "train_loss": tot / max(n, 1), **v})
        if verbose:
            print(f"epoch {ep+1:>4}  train {tot/max(n,1):.4f}  "
                  f"val MAE {v['mae']:.4f}  RMSE {v['rmse']:.4f}  "
                  f"R2 {v['r2']:.4f}  spread {v['std']:.4f}", flush=True)
        if v["mae"] < best - 1e-6:
            best, bad = v["mae"], 0
            if out:
                model.save(out)
        else:
            bad += 1
            if patience and bad >= patience:
                if verbose:
                    print(f"early stop at epoch {ep+1} (no val gain in {bad})")
                break

    if out and os.path.exists(out):
        model = PropertyGNN.load(out, device=device)
    test = _evaluate(model, dl_te, device)
    stats = {"best_val_mae": best, "test": test, "history": history}
    if verbose:
        print(f"best val MAE {best:.4f}  ->  {out}")
        print(f"HELD-OUT TEST  MAE {test['mae']:.4f}  RMSE {test['rmse']:.4f}  "
              f"R2 {test['r2']:.4f}  (n={test['n']})")
    if out:
        with open(os.path.splitext(out)[0] + ".train.json", "w") as f:
            json.dump(stats, f, indent=2)
    return stats
