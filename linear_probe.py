"""
Linear probe on frozen backbone embeddings (aquarium-list species, iNaturalist captive photos).

Frozen backbone -> image embedding (L2-normalized) -> multinomial logistic regression -> species
Train/test split by aquarium/location group (StratifiedGroupKFold), so no aquarium appears in both.

Usage:
    python linear_probe.py                              # DINOv2 ViT-S/14 (default)
    python linear_probe.py --backbone siglip2_so400m    # any key in backbones.BACKBONES
"""

import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import analyze_dinov2 as A
from backbones import BACKBONES, DEFAULT_BACKBONE, output_root

OUT = None  # set in main() from --backbone
N_FOLDS = 5
INNER_FOLDS = 3
C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
COLORS = A.COLORS


def load_data(backbone: str):
    """All scene-filtered photos of the aquarium-list species (no per-class equalization)."""
    species = A.prepare_inat_aquarium()
    scenes = A.scene_filter(A.AQUARIUM_DIR, species)
    keep = scenes.loc[~scenes["scene"].isin(A.SCENE_EXCLUDE), "image_path"]
    meta = pd.DataFrame({"image_path": keep})
    meta["species"] = meta["image_path"].map(lambda p: p.split("/")[-2].replace("_", " "))
    meta = meta[meta["species"].isin(species)].sort_values("image_path").reset_index(drop=True)

    cache_emb, cache_meta = OUT / "embeddings_all.npy", OUT / "metadata_all.csv"
    if cache_emb.exists() and cache_meta.exists():
        cached = pd.read_csv(cache_meta)
        if cached["image_path"].tolist() == meta["image_path"].tolist():
            print(f"  using cached embeddings: {cache_emb.relative_to(A.ROOT)}")
            return species, np.load(cache_emb), cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = A.load_model(device, backbone)
    emb, meta = A.extract_embeddings(bb.model, meta, device, bb.transforms["crop"], bb.amp, bb.batch)
    meta = A.inat_extra_meta(meta)
    sp_list = A.aquarium_list_species()
    tid = dict(zip(sp_list["inat_name"], sp_list["taxon_id"].astype(int)))
    families = A.inat_taxonomy(species, [tid[s] for s in species])
    meta["family"] = meta["species"].map(families)
    meta["place_group"] = A.inat_place_groups(meta)
    np.save(cache_emb, emb)
    meta.to_csv(cache_meta, index=False, encoding="utf-8-sig")
    return species, emb, meta


def fit_lr(X, y, C):
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return clf


def choose_C(X, y, groups):
    """Inner grouped CV on the training fold only."""
    inner = GroupKFold(n_splits=INNER_FOLDS)
    scores = {}
    for C in C_GRID:
        f1s = []
        for tr, va in inner.split(X, y, groups):
            clf = fit_lr(X[tr], y[tr], C)
            f1s.append(f1_score(y[va], clf.predict(X[va]), average="macro",
                                labels=np.unique(y[va])))
        scores[C] = float(np.mean(f1s))
    return max(scores, key=scores.get), scores


def run_cv(X, y, groups, n_classes, fam_of_class, label, fixed_C=None):
    print(f"\n[{label}] {N_FOLDS}-fold StratifiedGroupKFold ({len(set(groups))} groups)")
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=A.SEED)
    folds, y_true_all, y_pred_all, proba_all, idx_all = [], [], [], [], []
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        assert not set(groups[tr]) & set(groups[te])  # no group (aquarium) in both train and test
        missing = n_classes - len(np.unique(y[tr]))
        if fixed_C is None:
            C, inner_scores = choose_C(X[tr], y[tr], groups[tr])
        else:
            C, inner_scores = fixed_C, {}
        clf = fit_lr(X[tr], y[tr], C)
        proba = np.zeros((len(te), n_classes))
        proba[:, clf.classes_] = clf.predict_proba(X[te])
        pred = proba.argmax(1)
        labels_te = np.unique(y[te])
        m = {
            "fold": k, "C": C, "n_train": len(tr), "n_test": len(te),
            "test_groups": len(set(groups[te])), "test_classes": len(labels_te),
            "classes_missing_in_train": missing,
            "top1": float((pred == y[te]).mean()),
            "top5": float(top_k_accuracy_score(y[te], proba, k=5, labels=np.arange(n_classes))),
            "macro_f1": float(f1_score(y[te], pred, average="macro", labels=labels_te)),
            "balanced_acc": float(balanced_accuracy_score(y[te], pred)),
            "family_top1": float((fam_of_class[pred] == fam_of_class[y[te]]).mean()),
        }
        folds.append(m)
        print(f"  fold {k}: C={C:<5} train={len(tr):5d} test={len(te):5d} ({m['test_groups']} groups, "
              f"{m['test_classes']} spp)  top1={m['top1']:.3f} top5={m['top5']:.3f} "
              f"macroF1={m['macro_f1']:.3f} famTop1={m['family_top1']:.3f}", flush=True)
        y_true_all.append(y[te]); y_pred_all.append(pred); proba_all.append(proba); idx_all.append(te)
    return (pd.DataFrame(folds), np.concatenate(y_true_all), np.concatenate(y_pred_all),
            np.concatenate(proba_all), np.concatenate(idx_all))


def summarize(df):
    cols = ["top1", "top5", "macro_f1", "balanced_acc", "family_top1"]
    return {c: (float(df[c].mean()), float(df[c].std())) for c in cols}


def plot_per_species(per_sp, path, label):
    per_sp = per_sp.sort_values("f1")
    fams = per_sp["family"].value_counts().index[:7].tolist()
    colors = [COLORS[fams.index(f)] if f in fams else "#b5b5b5" for f in per_sp["family"]]
    fig, ax = plt.subplots(figsize=(10, 17), dpi=150)
    ax.barh(range(len(per_sp)), per_sp["f1"], color=colors, height=0.75)
    ax.set_yticks(range(len(per_sp)), [f"{s}  [{f}]" for s, f in zip(per_sp["species"], per_sp["family"])],
                  fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (pooled over 5 aquarium-grouped test folds)")
    ax.set_title(f"Linear probe on frozen {label} - per-species F1\n"
                 "iNaturalist captive photos, train/test split by aquarium", fontsize=11)
    ax.grid(True, axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in range(len(fams))]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#b5b5b5"))
    ax.legend(handles, fams + ["Other families"], loc="lower right", fontsize=8, title="Family")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y_true, y_pred, species, families, path, label):
    n = len(species)
    order = sorted(range(n), key=lambda i: (families[species[i]], species[i]))
    cm = np.zeros((n, n))
    np.add.at(cm, (y_true, y_pred), 1)
    cm = cm / cm.sum(1, keepdims=True)
    cm = cm[np.ix_(order, order)]
    names = [species[i] for i in order]
    fig, ax = plt.subplots(figsize=(16, 14.5), dpi=150)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), [A.short_name(x) for x in names], rotation=90, fontsize=4.5)
    ax.set_yticks(range(n), [f"{A.short_name(x)} [{families[x]}]" for x in names], fontsize=4.5)
    fam = [families[x] for x in names]
    for i in range(1, n):
        if fam[i] != fam[i - 1]:
            ax.axhline(i - 0.5, color="#e34948", linewidth=0.4)
            ax.axvline(i - 0.5, color="#e34948", linewidth=0.4)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title("Linear probe confusion matrix (row-normalized; ordered by family; red lines = family "
                 f"boundaries)\nfrozen {label}, train/test split by aquarium, 5 folds pooled", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="fraction of true-species test photos")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=list(BACKBONES), default=DEFAULT_BACKBONE)
    parser.add_argument("--c-grid", type=float, nargs="+", default=None,
                        help=f"candidate C values for the inner grouped CV (default {C_GRID})")
    args = parser.parse_args()
    if args.c_grid:
        C_GRID[:] = sorted(args.c_grid)
    label = BACKBONES[args.backbone]["label"]
    OUT = output_root(A.ROOT, args.backbone) / "inat_aquarium" / "linear_probe"
    OUT.mkdir(parents=True, exist_ok=True)
    out_rel = OUT.relative_to(A.ROOT).as_posix()
    print(f"Linear probe: frozen {label} + logistic regression, split by aquarium -> {out_rel}")
    species, emb, meta = load_data(args.backbone)
    X = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    y = meta["species"].map({s: i for i, s in enumerate(species)}).to_numpy()
    families = dict(zip(meta["species"], meta["family"]))
    fam_of_class = np.array([families[s] for s in species])
    counts = meta["species"].value_counts()
    print(f"  images: {len(meta)}  species: {len(species)}  families: {len(set(families.values()))}  "
          f"per species: {counts.min()}-{counts.max()} (median {int(counts.median())})")
    print(f"  aquarium/location groups: {meta['place_group'].nunique()}")

    # Main: split by aquarium/location group
    df_place, yt, yp, proba, idx = run_cv(X, y, meta["place_group"].to_numpy(), len(species),
                                          fam_of_class, "split by aquarium")
    # Reference: split by observation only (same aquarium may appear in train and test -> optimistic).
    # Uses the C most often chosen in the aquarium split, so only the split differs.
    ref_C = float(df_place["C"].mode().iloc[0])
    df_obs, *_ = run_cv(X, y, meta["observation_id"].to_numpy(), len(species), fam_of_class,
                        f"reference: split by observation only (C={ref_C:g})", fixed_C=ref_C)

    s_place, s_obs = summarize(df_place), summarize(df_obs)

    # Per-species results (pooled over folds, aquarium split)
    f1s = f1_score(yt, yp, average=None, labels=np.arange(len(species)), zero_division=0)
    top5_hit = (np.argsort(-proba, axis=1)[:, :5] == yt[:, None]).any(1)
    per_sp = pd.DataFrame({
        "species": species, "family": fam_of_class,
        "common_name": [meta.loc[meta["species"] == s, "korean_name"].iloc[0] for s in species],
        "n_photos": [int(counts[s]) for s in species],
        "n_groups": [meta.loc[meta["species"] == s, "place_group"].nunique() for s in species],
        "f1": f1s,
        "recall_top1": [float((yp[yt == k] == k).mean()) for k in range(len(species))],
        "recall_top5": [float(top5_hit[yt == k].mean()) for k in range(len(species))],
    })
    per_sp.sort_values("f1").to_csv(OUT / "per_species_results.csv", index=False, encoding="utf-8-sig")

    cm = np.zeros((len(species), len(species)))
    np.add.at(cm, (yt, yp), 1)
    rown = cm / cm.sum(1, keepdims=True)
    pairs = sorted(((rown[i, j], i, j) for i in range(len(species)) for j in range(len(species)) if i != j),
                   reverse=True)[:20]
    wrong = yp != yt
    same_fam_err = float((fam_of_class[yp][wrong] == fam_of_class[yt][wrong]).mean())

    fam_df = per_sp.groupby("family").agg(n_species=("species", "size"), mean_f1=("f1", "mean")) \
        .sort_values("mean_f1")
    pred_df = meta.iloc[idx][["image_path", "species", "family", "place_group", "observation_url"]].copy()
    pred_df["predicted"] = [species[i] for i in yp]
    pred_df["top5"] = ["; ".join(species[j] for j in np.argsort(-p)[:5]) for p in proba]
    pred_df["correct"] = pred_df["predicted"] == pred_df["species"]
    pred_df.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    df_place.to_csv(OUT / "folds_split_by_aquarium.csv", index=False)
    df_obs.to_csv(OUT / "folds_split_by_observation.csv", index=False)

    plot_per_species(per_sp, OUT / "per_species_f1.png", label)
    plot_confusion(yt, yp, species, families, OUT / "confusion_matrix.png", label)

    fmt = lambda s, k: f"{s[k][0]:.3f} ± {s[k][1]:.3f}"
    bar = "=" * 48
    lines = [
        bar, "", f"{label} Linear Probe - Aquarium species (iNaturalist captive)", "", bar, "",
        "Model:", f"{label} (frozen) -> {emb.shape[1]}-d embedding (L2-normalized) -> logistic regression "
        "(multinomial, class_weight=balanced)", "",
        "Data:", f"{len(meta)} photos, {len(species)} species, {len(set(families.values()))} families, "
        f"{counts.min()}-{counts.max()} photos/species", "",
        "Split:", f"{N_FOLDS}-fold StratifiedGroupKFold by aquarium/location group "
        f"({meta['place_group'].nunique()} groups); C chosen by inner {INNER_FOLDS}-fold grouped CV "
        f"on the training fold only (grid {C_GRID})", "",
        "Results (mean ± std over folds)       split by AQUARIUM      | ref: split by observation",
        f"  Top-1 accuracy                      {fmt(s_place, 'top1')}          | {fmt(s_obs, 'top1')}",
        f"  Top-5 accuracy                      {fmt(s_place, 'top5')}          | {fmt(s_obs, 'top5')}",
        f"  Macro-F1                            {fmt(s_place, 'macro_f1')}          | {fmt(s_obs, 'macro_f1')}",
        f"  Balanced accuracy                   {fmt(s_place, 'balanced_acc')}          | "
        f"{fmt(s_obs, 'balanced_acc')}",
        f"  Family-level top-1                  {fmt(s_place, 'family_top1')}          | "
        f"{fmt(s_obs, 'family_top1')}",
        "", f"Chosen C per fold: {df_place['C'].tolist()}",
        f"Share of errors within the same family: {same_fam_err:.3f}",
        f"Species with F1 >= 0.9: {(per_sp['f1'] >= 0.9).sum()}, >= 0.8: {(per_sp['f1'] >= 0.8).sum()}, "
        f"< 0.6: {(per_sp['f1'] < 0.6).sum()}", "",
        "Worst 15 species (F1, top-1 recall, top-5 recall):",
        *[f"  {r.f1:.3f}  {r.recall_top1:.3f}  {r.recall_top5:.3f}  {r.species} [{r.family}]"
          for r in per_sp.sort_values("f1").head(15).itertuples()], "",
        "Best 10 species (F1):",
        *[f"  {r.f1:.3f}  {r.species} [{r.family}]" for r in per_sp.sort_values("f1", ascending=False)
          .head(10).itertuples()], "",
        "Top-20 confusions (fraction of true-species test photos predicted as the other species):",
        *[f"  {v:.3f}  {species[i]} [{fam_of_class[i]}] -> {species[j]} [{fam_of_class[j]}]"
          for v, i, j in pairs], "",
        "Families with >= 3 species (mean F1):",
        fam_df[fam_df["n_species"] >= 3].round(3).to_string(), "",
        "Output:",
        *[f"  {out_rel}/{f}" for f in ["per_species_f1.png", "confusion_matrix.png", "per_species_results.csv",
                                       "test_predictions.csv", "folds_split_by_aquarium.csv",
                                       "folds_split_by_observation.csv"]],
        "", bar,
    ]
    print("\n" + "\n".join(lines))
    (OUT / "linear_probe_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "linear_probe_results.json").write_text(json.dumps(
        {"backbone": args.backbone, "backbone_label": label, "embedding_dim": int(emb.shape[1]),
         "n_images": len(meta), "c_grid": list(C_GRID), "chosen_C": df_place["C"].tolist(),
         "split_by_aquarium": s_place, "split_by_observation": s_obs,
         "same_family_error_share": same_fam_err,
         "species_f1_ge_0.9": int((per_sp["f1"] >= 0.9).sum()),
         "species_f1_ge_0.8": int((per_sp["f1"] >= 0.8).sum()),
         "species_f1_lt_0.6": int((per_sp["f1"] < 0.6).sum()),
         "family_mean_f1": fam_df[fam_df["n_species"] >= 3]["mean_f1"].round(4).to_dict()},
        indent=2), encoding="utf-8")
    print(f"Saved: {out_rel}/linear_probe_summary.txt")


if __name__ == "__main__":
    main()
