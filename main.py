"""
Main experimental driver for Time-Series Forecasting.
Executes baseline models and 5 controlled research experiments:
  - Exp 1: Architectural benchmark (Vanilla RNN vs. LSTM vs. GRU vs. Persistence Baseline)
  - Exp 2: Sequence length sensitivity (w in [7, 14, 30])
  - Exp 3: Hidden capacity sensitivity (hidden_size in [32, 64, 128])
  - Exp 4: Regularization study (dropout in [0.0, 0.2, 0.4])
  - Exp 5: Efficiency trade-off (training time, inference latency, parameter count vs. RMSE)
Generates publication-quality charts and saves reproducible metrics.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import (
    CHECKPOINTS_DIR,
    DatasetConfig,
    METRICS_DIR,
    ModelConfig,
    PLOTS_DIR,
    RAW_DATA_DIR,
    TrainingConfig,
)
from models import build_model
from src.data_loader import BANK_NAMES, get_dataset_info, get_series, load_raw_data
from src.dataset import create_dataloaders
from src.evaluate import evaluate_forecast, predict_model
from src.preprocessing import persistence_baseline, prepare_datasets_pipeline
from src.train import train_model
from src.utils import (
    count_parameters,
    plot_comparison_bar,
    plot_loss_curves,
    plot_multiseries_comparison,
    plot_predictions,
    set_seed,
)


def run_experiment_1_model_comparison(
    series: np.ndarray,
    dataset_name: str,
    device: str = "cpu",
    epochs: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, Dict[str, Dict[str, List[float]]]]:
    """
    Experiment 1: Head-to-head comparison of Vanilla RNN, LSTM, and GRU alongside Persistence Baseline.
    """
    print(f"\n{'='*70}\n[EXPERIMENT 1] Architecture Comparison on {dataset_name}\n{'='*70}")
    set_seed(42)

    seq_len = 30
    forecast_horizon = 1
    hidden_size = 64

    # Prepare datasets
    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), scaler = prepare_datasets_pipeline(
        series, sequence_length=seq_len, forecast_horizon=forecast_horizon, test_samples=251
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), batch_size=32
    )

    models_to_test = ["rnn", "lstm", "gru"]
    results: List[Dict] = []
    test_predictions: Dict[str, np.ndarray] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}

    # 1. Baseline Evaluation
    baseline_pred_scaled = persistence_baseline(test_X, forecast_horizon=forecast_horizon)
    base_metrics = evaluate_forecast(test_Y, baseline_pred_scaled, scaler=scaler)
    base_pred_orig = scaler.inverse_transform(baseline_pred_scaled)
    test_predictions["Persistence Baseline"] = base_pred_orig[:, 0]

    results.append({
        "Model": "Persistence Baseline",
        "Parameters": 0,
        "Train_Time_Sec": 0.0,
        "Best_Epoch": 0,
        "Scaled_MAE": base_metrics["scaled_MAE"],
        "Scaled_RMSE": base_metrics["scaled_RMSE"],
        "Scaled_MDA_Pct": base_metrics["scaled_MDA_Pct"],
        "Unscaled_MAE": base_metrics["unscaled_MAE"],
        "Unscaled_RMSE": base_metrics["unscaled_RMSE"],
        "Unscaled_MDA_Pct": base_metrics["unscaled_MDA_Pct"],
    })

    # 2. Train and evaluate RNN, LSTM, GRU
    for m_name in models_to_test:
        print(f"\n--- Training {m_name.upper()} ---")
        set_seed(42)
        model = build_model(
            m_name,
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            output_size=forecast_horizon,
            dropout=0.0,
        )
        param_count = count_parameters(model)
        ckpt_path = CHECKPOINTS_DIR / f"exp1_{dataset_name.lower()}_{m_name}.pt"

        trained_model, history, train_time, best_epoch = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=0.001,
            patience=15,
            device=device,
            checkpoint_path=ckpt_path,
            verbose=False,
        )
        histories[m_name] = history

        # Inference
        y_true_scaled, y_pred_scaled = predict_model(trained_model, test_loader, device=device)
        metrics = evaluate_forecast(y_true_scaled, y_pred_scaled, scaler=scaler)

        y_pred_orig = scaler.inverse_transform(y_pred_scaled)
        display_name = "Vanilla RNN" if m_name == "rnn" else m_name.upper()
        test_predictions[display_name] = y_pred_orig[:, 0]

        results.append({
            "Model": display_name,
            "Parameters": param_count,
            "Train_Time_Sec": round(train_time, 2),
            "Best_Epoch": best_epoch,
            "Scaled_MAE": metrics["scaled_MAE"],
            "Scaled_RMSE": metrics["scaled_RMSE"],
            "Scaled_MDA_Pct": metrics["scaled_MDA_Pct"],
            "Unscaled_MAE": metrics["unscaled_MAE"],
            "Unscaled_RMSE": metrics["unscaled_RMSE"],
            "Unscaled_MDA_Pct": metrics["unscaled_MDA_Pct"],
        })
        print(
            f"Result [{display_name}]: Unscaled RMSE={metrics['unscaled_RMSE']:.4f}, "
            f"Scaled RMSE={metrics['scaled_RMSE']:.6f}, MDA={metrics['unscaled_MDA_Pct']:.2f}%, Time={train_time:.2f}s"
        )

    y_test_orig = scaler.inverse_transform(test_Y)[:, 0]
    df_results = pd.DataFrame(results)
    return df_results, test_predictions, y_test_orig, histories


def run_experiment_2_sequence_lengths(
    series: np.ndarray,
    device: str = "cpu",
    epochs: int = 100,
) -> pd.DataFrame:
    """
    Experiment 2: Impact of input sequence lookback length (w in [7, 14, 30]).
    """
    print(f"\n{'='*70}\n[EXPERIMENT 2] Sequence Length Comparison (w = 7, 14, 30)\n{'='*70}")
    seq_lengths = [7, 14, 30]
    models = ["rnn", "lstm", "gru"]
    records = []

    for seq_len in seq_lengths:
        (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), scaler = prepare_datasets_pipeline(
            series, sequence_length=seq_len, forecast_horizon=1, test_samples=251
        )
        train_loader, val_loader, test_loader = create_dataloaders(
            (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), batch_size=32
        )

        for m_name in models:
            set_seed(42)
            model = build_model(m_name, input_size=1, hidden_size=64, num_layers=1, output_size=1)
            trained_model, _, train_time, best_epoch = train_model(
                model, train_loader, val_loader, epochs=epochs, lr=0.001, patience=15, device=device, verbose=False
            )
            y_true_scaled, y_pred_scaled = predict_model(trained_model, test_loader, device=device)
            metrics = evaluate_forecast(y_true_scaled, y_pred_scaled, scaler=scaler)

            display_name = "Vanilla RNN" if m_name == "rnn" else m_name.upper()
            records.append({
                "Sequence_Length": seq_len,
                "Model": display_name,
                "Unscaled_RMSE": metrics["unscaled_RMSE"],
                "Unscaled_MAE": metrics["unscaled_MAE"],
                "Scaled_RMSE": metrics["scaled_RMSE"],
                "MDA_Pct": metrics["unscaled_MDA_Pct"],
                "Train_Time_Sec": round(train_time, 2),
            })
            print(f"w={seq_len} | {display_name} -> Unscaled RMSE={metrics['unscaled_RMSE']:.4f}")

    return pd.DataFrame(records)


def run_experiment_3_hidden_sizes(
    series: np.ndarray,
    device: str = "cpu",
    epochs: int = 100,
) -> pd.DataFrame:
    """
    Experiment 3: Capacity and representational power across hidden sizes (32, 64, 128).
    """
    print(f"\n{'='*70}\n[EXPERIMENT 3] Hidden Unit Size Comparison (H = 32, 64, 128)\n{'='*70}")
    hidden_sizes = [32, 64, 128]
    models = ["rnn", "lstm", "gru"]
    records = []

    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), scaler = prepare_datasets_pipeline(
        series, sequence_length=30, forecast_horizon=1, test_samples=251
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), batch_size=32
    )

    for h_size in hidden_sizes:
        for m_name in models:
            set_seed(42)
            model = build_model(m_name, input_size=1, hidden_size=h_size, num_layers=1, output_size=1)
            params = count_parameters(model)
            trained_model, _, train_time, best_epoch = train_model(
                model, train_loader, val_loader, epochs=epochs, lr=0.001, patience=15, device=device, verbose=False
            )
            y_true_scaled, y_pred_scaled = predict_model(trained_model, test_loader, device=device)
            metrics = evaluate_forecast(y_true_scaled, y_pred_scaled, scaler=scaler)

            display_name = "Vanilla RNN" if m_name == "rnn" else m_name.upper()
            records.append({
                "Hidden_Size": h_size,
                "Model": display_name,
                "Parameters": params,
                "Unscaled_RMSE": metrics["unscaled_RMSE"],
                "Unscaled_MAE": metrics["unscaled_MAE"],
                "Scaled_RMSE": metrics["scaled_RMSE"],
                "MDA_Pct": metrics["unscaled_MDA_Pct"],
                "Train_Time_Sec": round(train_time, 2),
            })
            print(f"H={h_size} | {display_name} -> Unscaled RMSE={metrics['unscaled_RMSE']:.4f}, Params={params}")

    return pd.DataFrame(records)


def run_experiment_4_dropout(
    series: np.ndarray,
    device: str = "cpu",
    epochs: int = 100,
) -> pd.DataFrame:
    """
    Experiment 4: Effect of dropout regularization (p in [0.0, 0.2, 0.4]).
    """
    print(f"\n{'='*70}\n[EXPERIMENT 4] Dropout Regularization Study (p = 0.0, 0.2, 0.4)\n{'='*70}")
    dropouts = [0.0, 0.2, 0.4]
    models = ["rnn", "lstm", "gru"]
    records = []

    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), scaler = prepare_datasets_pipeline(
        series, sequence_length=30, forecast_horizon=1, test_samples=251
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), batch_size=32
    )

    for drop in dropouts:
        for m_name in models:
            set_seed(42)
            model = build_model(m_name, input_size=1, hidden_size=64, num_layers=1, output_size=1, dropout=drop)
            trained_model, _, train_time, best_epoch = train_model(
                model, train_loader, val_loader, epochs=epochs, lr=0.001, patience=15, device=device, verbose=False
            )
            y_true_scaled, y_pred_scaled = predict_model(trained_model, test_loader, device=device)
            metrics = evaluate_forecast(y_true_scaled, y_pred_scaled, scaler=scaler)

            display_name = "Vanilla RNN" if m_name == "rnn" else m_name.upper()
            records.append({
                "Dropout": drop,
                "Model": display_name,
                "Unscaled_RMSE": metrics["unscaled_RMSE"],
                "Unscaled_MAE": metrics["unscaled_MAE"],
                "Scaled_RMSE": metrics["scaled_RMSE"],
                "MDA_Pct": metrics["unscaled_MDA_Pct"],
                "Train_Time_Sec": round(train_time, 2),
            })
            print(f"Dropout={drop} | {display_name} -> Unscaled RMSE={metrics['unscaled_RMSE']:.4f}")

    return pd.DataFrame(records)


def run_experiment_5_efficiency(
    exp1_df: pd.DataFrame,
    series: np.ndarray,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Experiment 5: Model efficiency, inference latency, parameter footprint vs. accuracy.
    """
    print(f"\n{'='*70}\n[EXPERIMENT 5] Computational Efficiency & Latency Benchmark\n{'='*70}")
    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), scaler = prepare_datasets_pipeline(
        series, sequence_length=30, forecast_horizon=1, test_samples=251
    )
    _, _, test_loader = create_dataloaders(
        (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), batch_size=1
    )

    records = []
    models = ["rnn", "lstm", "gru"]

    for m_name in models:
        set_seed(42)
        model = build_model(m_name, input_size=1, hidden_size=64, num_layers=1, output_size=1)
        model.eval()
        dev = torch.device(device)
        model.to(dev)

        # Warmup
        dummy = torch.randn(1, 30, 1, device=dev)
        for _ in range(20):
            _ = model(dummy)

        # Measure single-sample inference latency over 200 iterations
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(200):
                _ = model(dummy)
        total_inf_time = time.perf_counter() - start
        avg_latency_ms = (total_inf_time / 200.0) * 1000.0

        display_name = "Vanilla RNN" if m_name == "rnn" else m_name.upper()
        # Retrieve training time and RMSE from Exp 1 results
        match = exp1_df[exp1_df["Model"] == display_name]
        train_time = match["Train_Time_Sec"].values[0] if not match.empty else 0.0
        rmse = match["Unscaled_RMSE"].values[0] if not match.empty else 0.0
        params = match["Parameters"].values[0] if not match.empty else count_parameters(model)

        records.append({
            "Model": display_name,
            "Parameters": params,
            "Train_Time_Sec": train_time,
            "Inference_Latency_ms": round(avg_latency_ms, 3),
            "Unscaled_RMSE": rmse,
        })
        print(f"{display_name}: Params={params}, Latency={avg_latency_ms:.3f} ms, TrainTime={train_time:.2f}s, RMSE={rmse:.4f}")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Run complete time-series neural network experiments.")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda' or 'mps')")
    parser.add_argument("--dataset", type=str, default="BANKEX", choices=["BANKEX", "activities"])
    args = parser.parse_args()

    # Load data
    file_path = RAW_DATA_DIR / f"{args.dataset}.csv"
    print(f"Loading raw dataset from {file_path}...")
    raw_data = load_raw_data(file_path)
    print("Dataset summary:", get_dataset_info(raw_data, args.dataset))

    series = get_series(raw_data, series_idx=0)
    print(f"Selected series 0: total length {len(series)}")

    # 1. Run Experiment 1 (Model comparison)
    exp1_df, test_preds, y_test_orig, histories = run_experiment_1_model_comparison(
        series, dataset_name=args.dataset, device=args.device, epochs=args.epochs
    )
    exp1_df.to_csv(METRICS_DIR / "exp1_model_comparison.csv", index=False)
    print("\nExperiment 1 Results Table:")
    print(exp1_df.to_string(index=False))

    # Save Visualizations for Exp 1
    # Loss curves for all models
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    m_keys = [("rnn", "Vanilla RNN"), ("lstm", "LSTM"), ("gru", "GRU")]
    for i, (k, label) in enumerate(m_keys):
        hist = histories[k]
        axes[i].plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"], label="Train Loss", color="#1f77b4")
        axes[i].plot(range(1, len(hist["val_loss"]) + 1), hist["val_loss"], label="Val Loss", color="#ff7f0e", linestyle="--")
        axes[i].set_title(f"{label} Loss Curves", fontweight="bold")
        axes[i].set_xlabel("Epochs")
        if i == 0:
            axes[i].set_ylabel("MSE Loss")
        axes[i].legend()
        axes[i].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "training_validation_loss_curves.png", dpi=300)
    plt.close(fig)

    # Actual vs Predicted Plot
    plot_predictions(
        y_test_orig,
        test_preds,
        title=f"Actual vs. Predicted Stock Prices on {args.dataset} (Test Horizon)",
        ylabel="Closing Price (INR)" if args.dataset == "BANKEX" else "Activity Value",
        save_path=PLOTS_DIR / "actual_vs_predicted_time_series.png",
        max_points=120,
    )

    # Bar chart of RMSE comparison
    plot_comparison_bar(
        exp1_df,
        metric="Unscaled_RMSE",
        x_col="Model",
        title=f"Architectural Benchmark (Unscaled RMSE on {args.dataset})",
        ylabel="RMSE (INR)" if args.dataset == "BANKEX" else "RMSE (Units)",
        save_path=PLOTS_DIR / "rnn_lstm_gru_performance_comparison.png",
    )

    # 2. Run Experiment 2 (Sequence lengths)
    exp2_df = run_experiment_2_sequence_lengths(series, device=args.device, epochs=args.epochs)
    exp2_df.to_csv(METRICS_DIR / "exp2_sequence_lengths.csv", index=False)
    plot_multiseries_comparison(
        exp2_df,
        x_col="Sequence_Length",
        y_col="Unscaled_RMSE",
        hue_col="Model",
        title="Impact of Sequence Lookback Window on Forecast RMSE",
        xlabel="Sequence Length (w)",
        ylabel="Unscaled RMSE (INR)" if args.dataset == "BANKEX" else "RMSE (Units)",
        save_path=PLOTS_DIR / "sequence_length_rmse_comparison.png",
    )

    # 3. Run Experiment 3 (Hidden unit sizes)
    exp3_df = run_experiment_3_hidden_sizes(series, device=args.device, epochs=args.epochs)
    exp3_df.to_csv(METRICS_DIR / "exp3_hidden_sizes.csv", index=False)
    plot_multiseries_comparison(
        exp3_df,
        x_col="Hidden_Size",
        y_col="Unscaled_RMSE",
        hue_col="Model",
        title="Impact of Hidden Layer Size on Forecast RMSE",
        xlabel="Hidden Units (H)",
        ylabel="Unscaled RMSE (INR)" if args.dataset == "BANKEX" else "RMSE (Units)",
        save_path=PLOTS_DIR / "hidden_size_comparison.png",
    )

    # 4. Run Experiment 4 (Dropout study)
    exp4_df = run_experiment_4_dropout(series, device=args.device, epochs=args.epochs)
    exp4_df.to_csv(METRICS_DIR / "exp4_dropout.csv", index=False)
    plot_multiseries_comparison(
        exp4_df,
        x_col="Dropout",
        y_col="Unscaled_RMSE",
        hue_col="Model",
        title="Impact of Dropout Regularization on Forecast RMSE",
        xlabel="Dropout Rate (p)",
        ylabel="Unscaled RMSE (INR)" if args.dataset == "BANKEX" else "RMSE (Units)",
        save_path=PLOTS_DIR / "dropout_impact_comparison.png",
    )

    # 5. Run Experiment 5 (Efficiency & latency benchmark)
    exp5_df = run_experiment_5_efficiency(exp1_df, series, device=args.device)
    exp5_df.to_csv(METRICS_DIR / "exp5_efficiency_benchmark.csv", index=False)
    plot_comparison_bar(
        exp5_df,
        metric="Train_Time_Sec",
        x_col="Model",
        title="Training Time Comparison Across Recurrent Architectures",
        ylabel="Total Training Time (Seconds)",
        save_path=PLOTS_DIR / "training_time_comparison.png",
    )

    print(f"\n{'='*70}\nAll experiments completed successfully!")
    print(f"Results saved to: {METRICS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

