"""Create the three preregistered S8-R3 audit figures."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "artifacts" / "s8r3"; FIG = OUT / "figures"

def rows(path):
    with path.open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    curve = rows(OUT / "training_curve.csv")
    plt.figure(figsize=(8, 5))
    for model, color in (("BC", "#2b8cbe"), ("DIFFUSION", "#d95f0e")):
        part = [r for r in curve if r["model"] == model]
        x = [float(r["effective_pass"]) for r in part]; y = [float(r["val_offline_loss"]) for r in part]
        plt.plot(x, y, label=model, color=color)
    plt.xlabel("Effective data pass"); plt.ylabel("VAL offline loss"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "offline_loss_vs_effective_pass.png", dpi=180); plt.close()
    closed = rows(OUT / "closed_loop_metrics.csv")
    plt.figure(figsize=(8, 5))
    for model, color in (("BC", "#2b8cbe"), ("DIFFUSION", "#d95f0e")):
        part = [r for r in closed if r["model"] == model]
        plt.plot([int(r["pass"]) for r in part], [float(r["success_rate"]) for r in part], "o-", label=model, color=color)
    plt.xlabel("Effective data pass"); plt.ylabel("VAL success rate"); plt.xticks([1, 5, 10, 20, 30]); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "closed_loop_success_vs_effective_pass.png", dpi=180); plt.close()
    family = rows(OUT / "family_metrics.csv")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, p in zip(axes, (1, 10, 30)):
        for model, color, offset in (("BC", "#2b8cbe", -.18), ("DIFFUSION", "#d95f0e", .18)):
            part = [r for r in family if int(r["pass"]) == p and r["model"] == model]
            vals = [float(r["success_rate"]) for r in part]
            x = list(range(len(part))); ax.bar([i + offset for i in x], vals, width=.34, label=model if p == 1 else None, color=color)
        axes[0].set_title(f"Pass {p}") if p == 1 else None
        ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.2)
    axes[-1].set_xticks(range(len([r for r in family if int(r["pass"]) == 1 and r["model"] == "BC"])))
    axes[-1].set_xticklabels([r["family"] for r in family if int(r["pass"]) == 1 and r["model"] == "BC"], rotation=35, ha="right")
    axes[0].legend(); plt.tight_layout(); plt.savefig(FIG / "family_success_at_passes.png", dpi=180); plt.close()
    print(f"wrote {len(list(FIG.glob('*.png')))} figures to {FIG}")

if __name__ == "__main__": main()
