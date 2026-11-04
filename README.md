# FragInv

Property-targeted molecular generation by direct optimisation over a fragment graph. A candidate is two tensors, which motif occupies each slot and which slots are bonded. The molecule is decoded discretely at every step and rebuilt differentiably with a straight-through estimator, so a frozen property predictor scores exactly the molecule that comes out. Nothing is trained during generation.

The motif vocabularies and the trained predictors are included, so the reported runs reproduce without retraining anything, and the molecules those runs produced are under `generated_molecules/`.

The synthetic accessibility score comes from the contributed `sascorer` that ships inside RDKit, which the code locates itself. A GPU is optional, `--device cuda`.

Everything the generation needs is in the repository:

```
fraginv/              data.py, design.py, main.py, metrics.py, model.py, pipeline.py, vocab.py
data/                 zinc.csv, qm9_gap.csv
runs/zinc/            vocab.json, logp_gin.pt
runs/qm9/             vocab.json, gap_gin.pt
generated_molecules/  the molecules reported in the paper
generate.sh           all five windows in one command
```

A reference set is required for every run. Novel-only ranking is part of the method rather than a flag, which is why `--reference` cannot be omitted: anything already in the reference set is refused at ranking time, so novelty of the reported set is one by construction. The two files under `data/` are the sets the shipped vocabularies and predictors were built from, and using a different ZINC or QM9 subset changes what counts as novel.

The logP benchmark scores success with the Crippen oracle, which is exact and computed on the spot. The gap benchmark has no cheap ground truth. Generation on QM9 therefore runs with `--oracle none` and reports no success column of its own, unless a second checkpoint is passed with `--eval-model` to score the output, which is how the reported numbers were obtained. The ground truth behind the released gap molecules is DFT rather than any network.

## Hyperparameters

These are the exact configuration from Optuna search:
| Hyperparameter | Flag | logP (ZINC250K) | Gap (QM9) |
| --- | --- | --- | --- |
| Slots | `--slots` | 13 | 8 |
| Learning rate | `--lr` | 0.0550332861741317 | 0.024480665535994692 |
| Heavy-atom hinge weight | `--w-size` | 0.12927165004463206 | 0.10516168267631766 |
| Motif-frequency prior | `--prior-beta` | 1.7525293458204356 | 0.42961987298408966 |
| Initialisation noise | `--init-noise` | 1.594862521138023 | 1.1734940756831918 |
| Target-aware bias | `--affinity-gamma` | 3.4790829708829802 | 2.508869476887604 |
| Batch carried between rounds | `--resume-frac` | 0.1883150824955844 | 0.2691886259138968 |

The rest is protocol, fixed per benchmark rather than tuned, and identical across every run of that benchmark:

| Setting | Flag | logP (ZINC250K) | Gap (QM9) |
| --- | --- | --- | --- |
| Optimisation steps per round | `--steps` | 250 | 300 |
| Batch | `--batch` | 48 | 48 |
| Edge threshold | `--edge-threshold` | 0.5 | 0.5 |
| Archive every k steps | `--archive-every` | 10 | 10 |
| Heavy-atom range | `--min-atoms`, `--max-atoms` | 12 to 34 | 7 to 9 |
| Molecules per scaffold | `--max-per-scaffold` | 5 | 1000 |
| Ions refused at admission | `--neutral-only` | off | on |
| Ground truth | `--oracle` | logp | none, with `--eval-model` |
| Molecules reported | `-n` | 1000 | 100 |
| In-window surplus chased | `--surplus` | 3 | 10 |
| Round cap | `--rounds` | 40 | 40 |
| Seed | `--seed` | 0 | 0 |

Three of those deserve a word. `-n` is how many molecules are reported

## Generating

A window whose output directory already holds `metrics.json` is skipped, so an interrupted run continues where it stopped, and `FORCE=1` redoes it.

Low logP:

```bash
python -m fraginv.main generate \
  --vocab runs/zinc --model runs/zinc/logp_gin.pt --reference data/zinc.csv \
  --oracle logp --device cpu \
  --lo -2.5 --hi -2.0 --out out/logp_low \
  --slots 13 --lr 0.0550332861741317 --w-size 0.12927165004463206 \
  --prior-beta 1.7525293458204356 --init-noise 1.594862521138023 \
  --affinity-gamma 3.4790829708829802 --resume-frac 0.1883150824955844 \
  --steps 250 --batch 48 --edge-threshold 0.5 --archive-every 10 \
  --min-atoms 12 --max-atoms 34 \
  --max-sa 99 --allow-pains --allow-brenk --max-per-scaffold 5 \
  --seed 0 -n 1000 --surplus 3 --rounds 40 \
  --full-metrics --trace 25
```

High logP is the same command with `--lo 5.0 --hi 5.5 --out out/logp_high`.

The 4.1 eV gap window:

```bash
python -m fraginv.main generate \
  --vocab runs/qm9 --model runs/qm9/gap_gin.pt --reference data/qm9_gap.csv \
  --oracle none --eval-model runs/qm9/gap_gcn.pt --device cpu \
  --lo 3.6 --hi 4.6 --out out/gap_4.1 \
  --slots 8 --lr 0.024480665535994692 --w-size 0.10516168267631766 \
  --prior-beta 0.42961987298408966 --init-noise 1.1734940756831918 \
  --affinity-gamma 2.508869476887604 --resume-frac 0.2691886259138968 \
  --steps 300 --batch 48 --edge-threshold 0.5 --archive-every 10 \
  --min-atoms 7 --max-atoms 9 --neutral-only \
  --max-sa 99 --allow-pains --allow-brenk --max-per-scaffold 1000 \
  --seed 0 -n 100 --surplus 10 --rounds 40 \
  --full-metrics --trace 25
```

The other two are `--lo 6.3 --hi 7.3 --out out/gap_6.8` and `--lo 8.8 --hi 9.8 --out out/gap_9.3`.

`--baseline` decodes from unoptimised random parameters first and prints the result. It is the control the optimised numbers have to beat, otherwise the gradient is doing nothing. `--trace 25` prints a progress line every 25 steps. Neither changes what is generated.

Each run writes four files into its output directory. `molecules.csv` is the reported set with each molecule's proxy prediction, SA and QED. `visited.csv` is every distinct molecule the search passed through. `metrics.json` holds the success rates, diversity and cost. `config.json` records the configuration that produced them.

The released molecules are in `generated_molecules/`

`logp_low.csv` and `logp_high.csv` are the reported sets for the two ZINC windows, each molecule with the proxy prediction it was ranked by. The three `dft_gap_*.csv` files are the QM9 molecules together with their DFT ground truth, one row per calculation: `smiles_in` is the generated molecule, `pred` the proxy prediction, `gap_eV` the HOMO-LUMO gap from a B3LYP/6-31G(2df,p) geometry optimisation in ORCA, `smiles_out` the structure perceived from the relaxed coordinates, and `intact` whether relaxation left the molecular graph unchanged. `finished` and `converged` are both true on the rows that count, since an unconverged job can still leave a parsable orbital block and its gap is not a result.
