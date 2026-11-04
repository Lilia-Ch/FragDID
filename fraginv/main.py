from __future__ import annotations

import argparse

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

from .data import prep_qm9, read_smiles, read_targets
from .design import DesignConfig
from .metrics import print_stats
from .model import DenseMolDataset, PropertyGNN, train_predictor
from .pipeline import ORACLES, generate, load_stack
from .vocab import Vocab, build, feature_dim

RDLogger.DisableLog("rdApp.*")

def cmd_prep_qm9(a):
    prep_qm9(a.out, source=a.source, limit=a.limit)

def cmd_vocab(a):
    build(read_smiles(a.smiles, a.column, a.limit), a.out,
          min_count=a.min_count, max_motif_atoms=a.max_motif_atoms,
          neutralize=a.neutralize)

def cmd_train(a):
    vocab = Vocab(a.vocab)
    print(vocab)

    smis = read_smiles(a.smiles, a.column, a.limit)
    ys = read_targets(a.smiles, a.target_column, a.limit) if a.target_column else None

    pairs = []
    for i, s in enumerate(smis):
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() > a.max_atoms:
            continue
        y = ys[i] if ys is not None else Crippen.MolLogP(mol)
        if y != y:
            continue
        pairs.append((mol, y))
    if not pairs:
        raise SystemExit("no usable molecules")

    ymean = sum(y for _, y in pairs) / len(pairs)
    ystd = (sum((y - ymean) ** 2 for _, y in pairs) / len(pairs)) ** 0.5 or 1.0
    readout = a.readout if a.readout != "auto" else ("mean" if a.target_column
                                                     else "sum")
    print(f"{len(pairs)} molecules, target mean {ymean:.3f} sd {ystd:.3f}, "
          f"readout={readout}, conv={a.conv}, heads={a.heads}")

    ds = DenseMolDataset(pairs, vocab.elements, a.max_atoms)
    print(f"featurised {len(ds)} molecules")

    model = PropertyGNN(feature_dim(vocab.elements), hidden=a.hidden,
                        layers=a.layers, readout=readout,
                        elements=vocab.elements, n_heads=a.heads, conv=a.conv,
                        mean=ymean, std=ystd)
    train_predictor(model, ds, epochs=a.epochs, batch=a.batch, lr=a.lr,
                    val_frac=a.val_frac, test_frac=a.test_frac,
                    device=a.device, out=a.out, seed=a.seed,
                    patience=a.patience)

def cmd_generate(a):
    vocab, model = load_stack(a.vocab, a.model, a.device)
    eval_model = PropertyGNN.load(a.eval_model, device=a.device) if a.eval_model else None

    cfg = DesignConfig(
        target=(a.lo, a.hi), n_slots=a.slots, batch=a.batch, steps=a.steps,
        lr=a.lr, edge_threshold=a.edge_threshold,
        min_atoms=a.min_atoms, max_atoms=a.max_atoms,
        w_size=a.w_size,
        max_rounds=a.rounds, resume_frac=a.resume_frac,
        archive_every=a.archive_every, prior_beta=a.prior_beta,
        init_noise=a.init_noise, seed=a.seed,
        max_sa=a.max_sa, max_ring=a.max_ring,
        allow_pains=a.allow_pains, allow_brenk=a.allow_brenk,
        max_per_scaffold=a.max_per_scaffold,
        neutral_only=a.neutral_only,
        oracle=ORACLES.get(a.oracle), rank_by=a.rank_by, trace=a.trace)

    print(f"\nmodel: conv={model.conv_name} layers={len(model.layers)} "
          f"heads={len(model.heads)} readout={model.readout}")
    print(f"vocab: {vocab}")
    print(f"ports: uniform over legal BRICS ports (not optimised)")
    if eval_model is not None:
        print(f"independent evaluator: conv={eval_model.conv_name} "
              f"layers={len(eval_model.layers)} (never used by the search)")

    res = generate(cfg, vocab, model, a.n, device=a.device,
                   affinity=a.affinity, eval_model=eval_model,
                   reference=read_smiles(a.reference),
                   oracle_name=a.oracle, out_dir=a.out,
                   affinity_gamma=a.affinity_gamma,
                   check=not a.no_check, baseline=a.baseline,
                   full_metrics=a.full_metrics,
                   n_stop=a.n_stop if a.n_stop else a.surplus * a.n)

    rows, stats = res["rows"], res["stats"]
    print(f"\nvisited {len(res['archive'])} molecules passing the quality gate, "
          f"kept {len(rows)}")
    print(f"wrote {a.out}/molecules.csv, visited.csv, metrics.json, config.json")
    print_stats(stats)
    if rows:
        print("\ntop molecules:")
        for r in rows[:8]:
            extra = f"  eval {r['eval']:6.2f}" if "eval" in r else ""
            print(f"  {r['smiles']:<44} pred {r['pred']:6.2f}{extra}  "
                  f"SA {r['sa']:.2f}  QED {r['qed']:.2f}")
def _search_args(g, *, steps, rounds, slots, batch, lr):
    g.add_argument("--slots", type=int, default=slots)
    g.add_argument("--batch", type=int, default=batch)
    g.add_argument("--steps", type=int, default=steps)
    g.add_argument("--rounds", type=int, default=rounds)
    g.add_argument("--lr", type=float, default=lr)
    g.add_argument("--min-atoms", type=int, default=12)
    g.add_argument("--max-atoms", type=int, default=38)
    g.add_argument("--max-sa", type=float, default=4.5)
    g.add_argument("--max-ring", type=int, default=7)
    g.add_argument("--max-per-scaffold", type=int, default=5)
    g.add_argument("--allow-pains", action="store_true")
    g.add_argument("--allow-brenk", action="store_true")
    g.add_argument("--reference", required=True,
                   help="training set; novel-only ranking is always on, and "
                        "this is also the reference for novelty and W1")
    g.add_argument("--affinity", default="model", choices=["model", "none"],
                   help="target-aware motif initialisation from the frozen "
                        "predictor itself; 'none' is the ablation")
    g.add_argument("--affinity-gamma", type=float, default=1.0)
    g.add_argument("--oracle", default="logp", choices=["logp", "qed", "none"])
    g.add_argument("--eval-model", default=None,
                   help="second checkpoint used only to score the output")
    g.add_argument("--device", default="cpu")


def _loop_args(g):
    """Search settings that are fixed per benchmark rather than tuned."""
    g.add_argument("--edge-threshold", type=float, default=0.5)
    g.add_argument("--w-size", type=float, default=0.05,
                   help="weight on the heavy-atom hinge; the other "
                        "loss term is the property itself")
    g.add_argument("--resume-frac", type=float, default=0.5)
    g.add_argument("--archive-every", type=int, default=5)
    g.add_argument("--prior-beta", type=float, default=1.0)
    g.add_argument("--init-noise", type=float, default=1.0)
    g.add_argument("--neutral-only", action="store_true",
                   help="refuse ions at admission; required on QM9, where a "
                        "charged molecule has no comparable reference gap")


def main(argv=None):
    p = argparse.ArgumentParser("fraginv")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("prep-qm9", help="download/convert QM9 to smiles,gap_ev")
    g.add_argument("--out", default="data/qm9_gap.csv")
    g.add_argument("--source", default=None, help="local qm9.csv instead of downloading")
    g.add_argument("--limit", type=int, default=None)
    g.set_defaults(func=cmd_prep_qm9)
    g = sub.add_parser("vocab", help="build the motif vocabulary")
    g.add_argument("--smiles", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--min-count", type=int, default=100)
    g.add_argument("--max-motif-atoms", type=int, default=18)
    g.add_argument("--column", default=None)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--neutralize", action="store_true", help="neutralize zwitterions before fragmentation " "(QM9 DFT benchmark; leave off for ZINC/logP)")
    g.set_defaults(func=cmd_vocab)
    g = sub.add_parser("train", help="train the property predictor")
    g.add_argument("--smiles", required=True)
    g.add_argument("--vocab", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--target-column", default=None,help="column of the label; omit to use RDKit Crippen logP")
    g.add_argument("--column", default=None)
    g.add_argument("--readout", default="auto", choices=["auto", "sum", "mean"])
    g.add_argument("--max-atoms", type=int, default=40)
    g.add_argument("--hidden", type=int, default=192)
    g.add_argument("--layers", type=int, default=4)
    g.add_argument("--heads", type=int, default=5)
    g.add_argument("--conv", default="gin", choices=["gin", "gcn", "gat"])
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--val-frac", type=float, default=0.1)
    g.add_argument("--test-frac", type=float, default=0.1)
    g.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu")
    g.set_defaults(func=cmd_train)
    g = sub.add_parser("generate", help="inverse design into a property window")
    g.add_argument("--vocab", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--lo", type=float, required=True)
    g.add_argument("--hi", type=float, required=True)
    g.add_argument("-n", type=int, default=1000, help="how many molecules to REPORT")
    g.add_argument("--surplus", type=int, default=1, help="search until the archive holds surplus x n in-window " "molecules, so the reported n can be selected from a " "real pool rather than from the first n found")
    g.add_argument("--n-stop", type=int, default=None, help="absolute in-window target, overriding --surplus; " "set it very high to always spend the full --rounds")
    _search_args(g, steps=300, rounds=40, slots=12, batch=48, lr=0.05)
    _loop_args(g)
    g.add_argument("--rank-by", default="pred", choices=["pred", "oracle"])
    g.add_argument("--trace", type=int, default=0, help="print every N steps")
    g.add_argument("--full-metrics", action="store_true", help="also report descriptors and W1 distances")
    g.add_argument("--baseline", action="store_true", help="report the no-gradient random control first")
    g.add_argument("--no-check", action="store_true", help="skip the assemble/decode consistency assertion")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
