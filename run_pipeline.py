"""Run the whole analysis and write reports/ (metrics, figures, predictions).

    python run_pipeline.py [--data Crop_recommendation.csv] [--out reports]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from crop_analysis import features, recommend, yield_model  # noqa: E402

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
TRUTH, SHAP_C = "#2a78d6", "#eb6834"   # categorical slots 1 and 2

def plot_shap_vs_truth(truth: pd.Series, shap_grouped: pd.Series, path: Path) -> None:
    order = list(shap_grouped.sort_values().index)
    y = range(len(order))
    h = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    tv = [truth.get(k, 0.0) for k in order]
    sv = [shap_grouped[k] for k in order]
    ax.barh([i + h / 2 + 0.02 for i in y], tv, height=h, color=TRUTH, label="Known formula term")
    ax.barh([i - h / 2 - 0.02 for i in y], sv, height=h, color=SHAP_C, label="XGBoost mean |SHAP|")
    for i, (a, b) in enumerate(zip(tv, sv, strict=True)):
        if a:
            ax.text(a + 0.15, i + h / 2 + 0.02, f"{a:.2f}", va="center", fontsize=8, color=INK)
        ax.text(b + 0.15, i - h / 2 - 0.02, f"{b:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), order, fontsize=9, color=INK)
    ax.set_xlabel("mean absolute contribution to predicted yield (yield units)", fontsize=9, color=MUTED)
    ax.set_title("Does SHAP recover the formula the target was built from?", loc="left",
                 fontsize=11, color=INK)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("Crop_recommendation.csv"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = features.add_features(features.load(args.data))

    # ---- 1. crop recommendation --------------------------------------
    table, rec = recommend.compare(df)
    print("\nCrop recommendation (22 classes)\n", table.to_string())
    best_name = table.index[0]
    example = {"Nitrogen": 90, "Phosphorous": 42, "Potassium": 43, "Temperature": 21.0,
               "Humidity": 82.0, "pH": 6.5, "Rainfall": 203.0}
    top3 = recommend.recommend(rec["models"][best_name], rec["encoder"], example)
    print(f"\nexample field {example}\n  -> {top3}")

    # ---- 2. synthetic yield regression + SHAP ------------------------
    y = yield_model.train(df)
    print("\nYield regression (held-out 20%)")
    print(json.dumps(y["test"], indent=2), "\nbest params", y["best_params"], "cv R2", y["cv_r2"])

    importance, explanation = yield_model.shap_importance(y["model"], y["X_test"])
    truth = yield_model.ground_truth_spread(df)
    shap_grouped = yield_model.grouped_shap(importance)
    plot_shap_vs_truth(truth, shap_grouped, args.out / "shap_vs_formula.png")

    import shap
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.gcf().savefig(args.out / "shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    # ---- 3. outputs --------------------------------------------------
    df["Expected_yield"] = y["model"].predict(df[yield_model.NUMERIC + yield_model.CATEGORICAL])
    df["Fertility_level"] = features.fertility_level(df["Total_Nutrients"])
    df.to_csv(args.out / "crop_predictions.csv", index=False)

    metrics = {
        "recommendation": table.reset_index().to_dict(orient="records"),
        "recommendation_example": {"input": example, "top3": top3, "model": best_name},
        "yield": {k: y[k] for k in ("best_params", "cv_r2", "test")},
        "shap_mean_abs": importance.round(4).to_dict(),
        "shap_grouped_vs_formula": {
            k: {"shap": round(shap_grouped[k], 4), "formula": round(truth.get(k, 0.0), 4)}
            for k in shap_grouped.index
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {args.out}/metrics.json, shap_vs_formula.png, shap_beeswarm.png, crop_predictions.csv")


if __name__ == "__main__":
    main()
