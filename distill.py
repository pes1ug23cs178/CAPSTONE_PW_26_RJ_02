"""M9 distillation entry point (T5.2). Same trainer as train.py; the `distill` block of the config
switches on the teacher terms (logit KL at T=4, z MSE, concept BCE, map MSE) and the student input
degradation. Usage: python distill.py --config configs/s3_kd_full.yaml --seed 42
"""
from train import main

if __name__ == "__main__":
    main()
