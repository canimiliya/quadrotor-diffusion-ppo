from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "s8r4"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def main():
    curve = read(OUT / "training_curve.csv")
    fig, ax = plt.subplots()
    for model, color in (("UNET", "#d62728"), ("DIFFUSION", "#1f77b4")):
        rows = [r for r in curve if r["model"] == model]
        ax.plot([float(r["effective_pass"]) for r in rows], [float(r["val_offline_loss"]) for r in rows], label=model, color=color)
    ax.set(xlabel="Effective pass", ylabel="VAL offline MSE x0"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "mlp_vs_unet_offline_loss.png", dpi=160); plt.close(fig)

    metrics = read(OUT / "closed_loop_metrics.csv")
    fig, ax = plt.subplots()
    for model, color in (("UNET", "#d62728"), ("DIFFUSION", "#1f77b4")):
        rows = [r for r in metrics if r["model"] == model]
        ax.plot([int(r["pass"]) for r in rows], [float(r["success_rate"]) * 100 for r in rows], "o-", label=model, color=color)
    ax.set(xlabel="Effective pass", ylabel="VAL success (%)"); ax.set_xticks([1, 5, 10, 20, 30]); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "mlp_vs_unet_success.png", dpi=160); plt.close(fig)

    family = read(OUT / "family_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, p in zip(axes, (1, 10, 30)):
        rows = [r for r in family if int(r["pass"]) == p and r["model"] in ("UNET", "DIFFUSION")]
        families = sorted(set(r["family"] for r in rows))
        for model, color in (("UNET", "#d62728"), ("DIFFUSION", "#1f77b4")):
            values = {r["family"]: float(r["success_rate"]) * 100 for r in rows if r["model"] == model}
            ax.plot(families, [values[x] for x in families], "o-", label=model, color=color)
        ax.set_title(f"pass {p}"); ax.tick_params(axis="x", rotation=65)
    axes[0].set_ylabel("VAL success (%)"); axes[-1].legend(); fig.tight_layout()
    fig.savefig(FIG / "mlp_vs_unet_family.png", dpi=160); plt.close(fig)

if __name__ == "__main__":
    main()
