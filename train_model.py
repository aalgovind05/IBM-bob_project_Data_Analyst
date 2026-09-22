"""Train and save the Amazon sales prediction model."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_pipeline import MODEL_DIR, load_data, save_artifacts, train_model


def main() -> None:
    data = load_data()
    model, metrics, comparison = train_model(data)
    save_artifacts(model, metrics)
    print(f"Dataset shape: {data.shape}")
    print(f"MAE: INR {metrics['MAE']:,.2f}")
    print(f"RMSE: INR {metrics['RMSE']:,.2f}")
    print(f"R2: {metrics['R2']:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(comparison["Actual"], comparison["Predicted"], alpha=0.45, color="#2f80ed")
    limits = [0, max(comparison.max())]
    axes[0].plot(limits, limits, "r--")
    axes[0].set(xlabel="Actual Amount (₹)", ylabel="Predicted Amount (₹)", title="Actual vs Predicted")
    residuals = comparison["Actual"] - comparison["Predicted"]
    axes[1].hist(residuals, bins=30, color="#7c5cd8", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set(xlabel="Residual (₹)", ylabel="Frequency", title="Residual Distribution")
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "diagnostics.png")
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
