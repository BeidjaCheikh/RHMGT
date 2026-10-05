# G0 Multi-Objective Tuning

Architecture frozen: full HERA-HGT + fhgnn_adaptive, hidden=256, heads=8, BCE.

Search space:
- lr: 5e-5, 1e-4, 2e-4
- weight_decay: 1e-4, 3e-4, 5e-4
- dropout: 0.10, 0.20, 0.30
- layers: 3, 4
- batch_size: 64, 128

Validation objectives: ACC, F1, AUROC, AUPR.
The tuner does not read the test split.

Run:

python scripts/tune_g0_multiobjective.py \
  --train-csv data/paper_split/hERGAT_train_df.csv \
  --val-csv data/paper_split/hERGAT_valid_df.csv \
  --out tuning/g0_multiobjective_final \
  --cache-dir .cache/hera_hgt_g0_multiobjective \
  --seed 2026 \
  --trials 36 \
  --epochs 100 \
  --patience 15 \
  --hidden-dim 256 \
  --heads 8 \
  --loss bce

Then inspect:
cat tuning/g0_multiobjective_final/recommendation.json
cat tuning/g0_multiobjective_final/pareto_front.json
