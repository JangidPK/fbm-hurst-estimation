"""

End-to-end pipeline for the Anomalous Hurst Coefficient Prediction project.

Runs, in order:


    1. Generate synthetic fBm dataset
    2. Visualize sample trajectories
    3. Train baseline (Random Forest on features)
    4. Train LSTM
    5. Train Transformer
    6. Evaluate all models (global + sliced by H range)
    7. Produce all Phase-9 plots
    8. Print final comparison table

Usage:
    python main.py --n_samples 3000 --T 256 --epochs 60
"""

import argparse
import os

import numpy as np
import torch

from data.generator import FractionalBrownianMotionGenerator
from data.dataset import FBmDataset, train_val_test_split
from models.baseline import BaselineHurstRegressor
from models.lstm import LSTMHurstRegressor
from models.transformer import TransformerHurstRegressor
from training.trainer import train_model, predict
from evaluation.metrics import summarize, slice_metrics
from evaluation.plots import (
    plot_sample_paths,
    plot_loss_curves,
    plot_prediction_scatter,
    plot_error_vs_H,
    plot_residual_distribution,
)


def parse_args():
    p = argparse.ArgumentParser(description="fBm Hurst exponent prediction pipeline")
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="outputs")
    return p.parse_args()


def main():

    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Generate dataset
    # ------------------------------------------------------------------
    print(f"[1/8] Generating {args.n_samples} fBm trajectories of length {args.T}...")
    gen = FractionalBrownianMotionGenerator(T=args.T, method="davies-harte", seed=args.seed)
    X, y = gen.generate_dataset(n_samples=args.n_samples, H_range=(0.1, 0.9))

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = train_val_test_split(
        X, y, val_frac=0.15, test_frac=0.15, seed=args.seed
    )
    print(f"  Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}")

    # ------------------------------------------------------------------
    # 2. Visualize sample trajectories
    # ------------------------------------------------------------------
    print("[2/8] Plotting sample trajectories...")
    plot_sample_paths(gen, save_path=os.path.join(args.outdir, "sample_trajectories.png"))

    results = {}

    # ------------------------------------------------------------------
    # 3. Baseline model
    # ------------------------------------------------------------------

    print("[3/8] Training baseline (Random Forest on handcrafted features)...")

    baseline = BaselineHurstRegressor(kind="random_forest")
    baseline.fit(X_tr, y_tr)
    base_preds = baseline.predict(X_te)
    results["Linear/RF Baseline"] = (base_preds, y_te)
    print("  Baseline test metrics:", summarize(y_te, base_preds))

    # -----------=========----------------------------------------
    # 4. LSTM
    # ------------------------------------------------------------------

    print("[4/8] Training LSTM...")
    
    train_ds = FBmDataset(X_tr, y_tr, normalize="zscore")
    val_ds = FBmDataset(X_val, y_val, normalize="zscore")
    test_ds = FBmDataset(X_te, y_te, normalize="zscore")

    lstm = LSTMHurstRegressor(hidden_size=128, num_layers=2, dropout=0.2)
    lstm, lstm_history = train_model(
        lstm, train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    lstm_preds, lstm_targets = predict(lstm, test_ds)
    results["LSTM"] = (lstm_preds, lstm_targets)
    print("  LSTM test metrics:", summarize(lstm_targets, lstm_preds))

    plot_loss_curves(
        lstm_history["train_loss"], lstm_history["val_loss"],
        title="LSTM Training Curve",
        save_path=os.path.join(args.outdir, "lstm_loss_curve.png"),
    )

    # ------------------------------------------------------------------
    # 5. Transformer
    # ------------------------------------------------------------------
    print("[5/8] Training Transformer...")
    transformer = TransformerHurstRegressor(
        d_model=64, nhead=4, num_layers=3, dim_feedforward=256,
        dropout=0.1, use_cls_token=True, max_len=args.T,
    )
    transformer, tr_history = train_model(
        transformer, train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    tr_preds, tr_targets = predict(transformer, test_ds)
    results["Transformer"] = (tr_preds, tr_targets)
    print("  Transformer test metrics:", summarize(tr_targets, tr_preds))

    plot_loss_curves(
        tr_history["train_loss"], tr_history["val_loss"],
        title="Transformer Training Curve",
        save_path=os.path.join(args.outdir, "transformer_loss_curve.png"),
    )

    # ------------------------------------------------------------------
    # 6. Evaluation (global + sliced)
    # ------------------------------------------------------------------
    print("[6/8] Evaluating all models...")
    for name, (preds, targets) in results.items():
        print(f"\n--- {name} ---")
        print("Global:", summarize(targets, preds))
        print("By H range:")
        for slice_name, m in slice_metrics(targets, preds).items():
            print(f"  {slice_name}: {m}")

    # ------------------------------------------------------------------
    # 7. Plots (best deep model = lowest MAE among LSTM/Transformer)
    # ------------------------------------------------------------------
    print("[7/8] Producing prediction/error plots...")

    plot_prediction_scatter(
        lstm_targets, lstm_preds, title="LSTM: H_true vs H_pred",
        save_path=os.path.join(args.outdir, "lstm_scatter.png"),
    )

    plot_prediction_scatter(
        tr_targets, tr_preds, title="Transformer: H_true vs H_pred",
        save_path=os.path.join(args.outdir, "transformer_scatter.png"),
    )

    plot_error_vs_H(
        lstm_targets, lstm_preds, title="LSTM: Error vs H",
        save_path=os.path.join(args.outdir, "lstm_error_vs_H.png"),
    )


    plot_error_vs_H(
        tr_targets, tr_preds, title="Transformer: Error vs H",
        save_path=os.path.join(args.outdir, "transformer_error_vs_H.png"),
    )



    plot_residual_distribution(
        lstm_targets, lstm_preds, title="LSTM Residuals",
        save_path=os.path.join(args.outdir, "lstm_residuals.png"),
    )

    plot_residual_distribution(
        tr_targets, tr_preds, title="Transformer Residuals",
        save_path=os.path.join(args.outdir, "transformer_residuals.png"),
    )

    # ------------------------------------------------------------------
    # 8. Final comparison table
    # ------------------------------------------------------------------

    
    print("\n[8/8] Final comparison table")
    print(f"{'Model':<20}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for name, (preds, targets) in results.items():
        m = summarize(targets, preds)
        print(f"{name:<20}{m['MAE']:>10.4f}{m['RMSE']:>10.4f}{m['R2']:>10.4f}")

    print(f"\nAll plots saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
