"""
Collect metrics.json / linear_probe_results.json of every backbone into comparison tables and plots.

Usage:
    python compare_backbones.py [backbone ...]
"""

import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backbones import BACKBONES, output_root
from analyze_dinov2 import COLORS, ROOT

KEYS = sys.argv[1:] or ["dinov2_vits14", "eva02_large", "siglip2_so400m", "siglip2_giant", "pe_core_g14"]
OUT = ROOT / "output" / "backbone_comparison"

ANALYSES = [  # (name, path relative to the backbone's output root)
    ("iNat aquarium 100spp", "inat_aquarium/aquarium_only/metrics.json"),
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def metrics_from_summary(path):
    """Older runs (before metrics.json existed): read the same numbers from analysis_summary.txt."""
    if not path.exists():
        return None
    t = path.read_text(encoding="utf-8")
    sil = re.search(r"Silhouette Score:\n([\d.]+)\s+\(cosine, (\d+)-d", t)
    knn = re.findall(r"(\d\.\d+)\s+\((.*?excluded.*?)\)\n", t.split("10-NN Agreement")[1].split("PCA")[0])
    fam = re.findall(r"Family-level agreement \(species vote mapped to family\): ([\d.]+)", t)
    n = re.search(r"Total:\n(\d+)", t)
    out = {"embedding_dim": int(sil.group(2)), "n_images": int(n.group(1)),
           "silhouette_cosine": float(sil.group(1)), "nearest_centroid_loo_mean": None, "knn": []}
    for i, (acc, desc) in enumerate(knn):
        d = {"exclusion": desc, "knn_agreement": float(acc)}
        if i < len(fam):
            d["family_agreement"] = float(fam[i])
        out["knn"].append(d)
    stats = path.with_name("per_species_stats.csv")
    if stats.exists():
        out["nearest_centroid_loo_mean"] = float(pd.read_csv(stats)["nearest_centroid_is_own(LOO)"].mean())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {k: BACKBONES[k]["label"] for k in KEYS}

    # Feature analysis (no training)
    rows = []
    for k in KEYS:
        for name, rel in ANALYSES:
            m = load(output_root(ROOT, k) / rel) or metrics_from_summary(
                (output_root(ROOT, k) / rel).with_name("analysis_summary.txt"))
            if not m:
                print(f"  missing: {k} {rel}")
                continue
            knn = {("strict" if "STRICT" in x["exclusion"] else "basic"): x for x in m["knn"]}
            rows.append({
                "backbone": labels[k], "dim": m["embedding_dim"], "dataset": name, "images": m["n_images"],
                "silhouette_cos": m["silhouette_cosine"],
                "knn_basic": knn.get("basic", {}).get("knn_agreement"),
                "knn_strict": knn.get("strict", {}).get("knn_agreement"),
                "knn_strict_family": knn.get("strict", {}).get("family_agreement"),
                "nearest_centroid_loo": m["nearest_centroid_loo_mean"],
            })
    fa = pd.DataFrame(rows)
    fa.to_csv(OUT / "feature_analysis_comparison.csv", index=False)

    # Linear probe
    lp_rows, fam_rows, per_species = [], [], {}
    for k in KEYS:
        lp_dir = output_root(ROOT, k) / "inat_aquarium" / "linear_probe"
        m = load(lp_dir / "linear_probe_results.json")
        if not m:
            print(f"  missing linear probe: {k}")
            continue
        a, o = m["split_by_aquarium"], m["split_by_observation"]
        ps = pd.read_csv(lp_dir / "per_species_results.csv")
        dim = m.get("embedding_dim") or int(np.load(lp_dir / "embeddings_all.npy", mmap_mode="r").shape[1])
        fam_f1 = m.get("family_mean_f1") or ps.groupby("family").filter(lambda g: len(g) >= 3) \
            .groupby("family")["f1"].mean().to_dict()
        lp_rows.append({
            "backbone": labels[k], "dim": dim,
            "top1": a["top1"][0], "top1_std": a["top1"][1], "top5": a["top5"][0], "top5_std": a["top5"][1],
            "macro_f1": a["macro_f1"][0], "macro_f1_std": a["macro_f1"][1],
            "balanced_acc": a["balanced_acc"][0], "family_top1": a["family_top1"][0],
            "obs_split_top1": o["top1"][0],
            "chosen_C": m.get("chosen_C") or pd.read_csv(lp_dir / "folds_split_by_aquarium.csv")["C"].tolist(),
            "species_f1_ge_0.9": int((ps["f1"] >= 0.9).sum()), "species_f1_ge_0.8": int((ps["f1"] >= 0.8).sum()),
            "species_f1_lt_0.6": int((ps["f1"] < 0.6).sum()),
            "same_family_error_share": m["same_family_error_share"],
        })
        for fam, f1 in fam_f1.items():
            fam_rows.append({"backbone": labels[k], "family": fam, "mean_f1": f1})
        per_species[labels[k]] = ps.set_index("species")[["family", "f1", "recall_top5"]]
    lp = pd.DataFrame(lp_rows)
    lp.to_csv(OUT / "linear_probe_comparison.csv", index=False)
    fam = pd.DataFrame(fam_rows).pivot(index="family", columns="backbone", values="mean_f1")
    fam = fam[[labels[k] for k in KEYS if labels[k] in fam.columns]]
    fam.to_csv(OUT / "family_f1_comparison.csv")

    # Per-species F1 side by side
    names = list(per_species)
    if names:
        ps_all = pd.concat({n: per_species[n]["f1"] for n in names}, axis=1)
        ps_all.insert(0, "family", per_species[names[0]]["family"])
        ps_all["best"] = ps_all[names].idxmax(axis=1)
        ps_all.sort_values(names[0]).to_csv(OUT / "per_species_f1_comparison.csv")

    # Plot: linear probe headline metrics (grouped bars + std whiskers)
    if len(lp):
        metrics = [("top1", "Top-1"), ("top5", "Top-5"), ("macro_f1", "Macro-F1"), ("family_top1", "Family Top-1")]
        x = np.arange(len(metrics))
        w = 0.8 / len(lp)
        fig, ax = plt.subplots(figsize=(11, 5.4), dpi=150)
        for i, r in enumerate(lp.itertuples()):
            vals = [getattr(r, m) for m, _ in metrics]
            errs = [getattr(r, m + "_std", 0) if hasattr(r, m + "_std") else 0 for m, _ in metrics]
            bars = ax.bar(x + (i - (len(lp) - 1) / 2) * w, vals, w * 0.92, yerr=errs, capsize=3,
                          color=COLORS[i], label=f"{r.backbone} ({r.dim}-d)")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6, color="#222222", rotation=90)
        ax.set_xticks(x, [n for _, n in metrics])
        ax.set_ylim(0.6, 1.06)
        ax.set_ylabel("score (5-fold mean, split by aquarium)")
        ax.set_title("Linear probe on frozen backbones - 100 aquarium species\n"
                     "iNaturalist captive photos, train/test split by aquarium", fontsize=11)
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "linear_probe_comparison.png")
        plt.close(fig)

    print("\nFeature analysis (no training):")
    print(fa.round(3).to_string(index=False))
    print("\nLinear probe (100 species, split by aquarium):")
    print(lp.round(3).to_string(index=False))
    print("\nFamily mean F1 (families with >= 3 species):")
    print(fam.round(3).to_string())
    print(f"\nSaved to {OUT.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
