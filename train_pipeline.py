"""
Cyber Attack Detection Using Ensemble Learning
================================================
Trains and compares a baseline Decision Tree against three ensemble
approaches (Random Forest, XGBoost, soft-voting ensemble) on the official
UNSW-NB15 train/test split.

Tasks:
    binary            Normal vs Attack                       (target: 'label')
    multiclass        10 classes: Normal + 9 attack types    (target: 'attack_cat')
    multiclass_smote  same 10 classes, but the TRAINING set is oversampled with
                      SMOTENC so rare attack types have at least SMOTE_FLOOR rows.
                      The test set is never resampled.
    binary_pca        binary task on PCA-reduced features (keeps 95% of variance)
    multiclass_pca    multi-class task on PCA-reduced features

Usage:
    python train_pipeline.py                            # all five tasks (default)
    python train_pipeline.py --task binary_pca          # binary task with PCA
    python train_pipeline.py --task multiclass_smote    # SMOTE multi-class only
    python train_pipeline.py --task multiclass          # multi-class only
    python train_pipeline.py --task binary              # binary only

Outputs (artifacts/):
    encoders.pkl, scaler.pkl, feature_names.pkl, X_test_raw_sample.csv   (shared)
    binary/      <model>.pkl, results_summary.csv
    multiclass/  <model>.pkl, results_summary.csv, per_class_<model>.csv,
                 confusion_matrix_<best>.png
    multiclass_smote/  same files as multiclass/
    binary_pca/, multiclass_pca/  same files, models trained on PCA features
    pca/         pca.pkl (not committed), explained_variance.csv,
                 top_features_per_component.csv, explained_variance.png
    label_encoder.pkl (shared)
"""

import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

TRAIN_PATH = "data/UNSW_NB15_training-set.csv"   # official UNSW-NB15 train partition
TEST_PATH = "data/UNSW_NB15_testing-set.csv"     # official UNSW-NB15 test partition
ARTIFACTS_DIR = Path("artifacts")
RANDOM_STATE = 42
CAT_COLS = ["proto", "service", "state"]
PCA_VARIANCE = 0.95    # keep as many components as needed to explain 95% of training variance
SMOTE_FLOOR = 10_000   # classes smaller than this are oversampled up to this size


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
def load_data():
    missing = [p for p in (TRAIN_PATH, TEST_PATH) if not Path(p).exists()]
    if missing:
        raise SystemExit(
            f"Dataset file(s) not found: {', '.join(missing)}\n"
            "Download the official UNSW-NB15 training/testing partition CSVs "
            "(https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place them in ./data/ - see README.")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows | {train_df.shape[1]} columns")
    print("\nAttack category counts (train):")
    print(train_df["attack_cat"].value_counts().to_string())
    print("\nAttack category counts (test):")
    print(test_df["attack_cat"].value_counts().to_string())
    return train_df, test_df


# ---------------------------------------------------------------------------
# 2. PREPROCESS (shared by both tasks)
# ---------------------------------------------------------------------------
def preprocess(train_df, test_df):
    """
    - Drop 'id' (row index). 'label' and 'attack_cat' are the two targets.
    - Label-encode proto/service/state. The encoder is fit on the UNION of
      train+test categories because the official test set has 'state' values
      ('CLO', 'ACC') never seen in training. This only builds a vocabulary and
      uses no target information, so it does not leak labels.
    - StandardScaler is fit ONLY on training data, then applied to test.
    - 'attack_cat' is integer-encoded (alphabetical) for the multi-class task;
      XGBoost requires classes numbered 0..K-1.
    """
    train_x = train_df.drop(columns=["id", "label", "attack_cat"]).copy()
    test_x = test_df.drop(columns=["id", "label", "attack_cat"]).copy()

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([train_x[col], test_x[col]], axis=0))
        train_x[col] = le.transform(train_x[col])
        test_x[col] = le.transform(test_x[col])
        encoders[col] = le

    feature_names = list(train_x.columns)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_x)
    X_test = scaler.transform(test_x)

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train_df["attack_cat"], test_df["attack_cat"]]))

    targets = {
        "binary": (train_df["label"].values, test_df["label"].values),
        "multiclass": (label_encoder.transform(train_df["attack_cat"]),
                       label_encoder.transform(test_df["attack_cat"])),
    }

    # Shared artifacts used by demo.py
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(feature_names, ARTIFACTS_DIR / "feature_names.pkl")
    test_df.drop(columns=["id"]).to_csv(ARTIFACTS_DIR / "X_test_raw_sample.csv", index=False)

    return X_train, X_test, targets, feature_names, label_encoder, train_x, scaler


# ---------------------------------------------------------------------------
# 2b. SMOTE (multi-class, training set only)
# ---------------------------------------------------------------------------
def apply_smote(train_x, y_train, scaler, class_names, floor=SMOTE_FLOOR):
    """
    Oversample every class smaller than `floor` up to `floor` rows. Larger classes
    are left untouched (a floor, not full balancing, keeps training time sane).

    - SMOTENC, not plain SMOTE: proto/service/state are integer category codes, and
      plain SMOTE would interpolate between codes and invent meaningless categories.
      SMOTENC gives synthetic rows a real category (the majority among neighbours).
    - Resampling happens on the UNSCALED encoded features, then the ORIGINAL scaler
      (fit on real training data only) is applied, so the saved scaler and the demo
      stay consistent with the non-SMOTE models.
    - Applied to the training set only. The test set is never resampled.
    """
    counts = pd.Series(y_train).value_counts()
    strategy = {int(c): floor for c, n in counts.items() if n < floor}
    cat_idx = [list(train_x.columns).index(c) for c in CAT_COLS]
    smote = SMOTENC(categorical_features=cat_idx, sampling_strategy=strategy,
                    k_neighbors=5, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(train_x, y_train)
    print(f"  SMOTENC: {len(y_train):,} -> {len(y_res):,} training rows")
    before = pd.Series(y_train).value_counts().sort_index()
    after = pd.Series(y_res).value_counts().sort_index()
    for c in before.index:
        print(f"    {class_names[c]:<15}{before[c]:>8,} -> {after[c]:>8,}")
    return scaler.transform(X_res), y_res


# ---------------------------------------------------------------------------
# 2c. PCA (dimensionality reduction)
# ---------------------------------------------------------------------------
def fit_pca(X_train, X_test, feature_names, variance=PCA_VARIANCE):
    """
    Fit PCA on the SCALED TRAINING data only, then transform train and test.
    n_components=0.95 keeps the fewest components explaining 95% of the training variance,
    chosen in advance rather than tuned on test results.
    Also saves the explained-variance table/plot and, for each component, the original
    features with the largest absolute loadings (argmax of |loading|, as in the literature).
    Caveat: proto/service/state are arbitrary integer codes, so components that load on
    them are not very interpretable (one-hot would be cleaner but adds ~150 columns).
    """
    pca = PCA(n_components=variance, svd_solver="full", random_state=RANDOM_STATE)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    k = pca.n_components_
    print(f"  PCA: {X_train.shape[1]} features -> {k} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% of variance)")

    out = ARTIFACTS_DIR / "pca"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, out / "pca.pkl")
    ev = pd.DataFrame({"component": np.arange(1, k + 1),
                       "explained_variance_ratio": pca.explained_variance_ratio_,
                       "cumulative": np.cumsum(pca.explained_variance_ratio_)})
    ev.to_csv(out / "explained_variance.csv", index=False)

    names = np.array(feature_names)
    rows = []
    for i, comp in enumerate(pca.components_):
        order = np.argsort(np.abs(comp))[::-1][:3]
        rows.append({"component": i + 1, "top_feature": names[order[0]],
                     "top3_features": ", ".join(names[order]),
                     "top_loading": round(float(comp[order[0]]), 3)})
    pd.DataFrame(rows).to_csv(out / "top_features_per_component.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ev["component"], ev["cumulative"], marker="o", ms=3, color="#1F3864")
    ax.axhline(variance, color="#C0504D", ls="--", lw=1)
    ax.axvline(k, color="#C0504D", ls="--", lw=1)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"PCA: {k} components keep {variance:.0%} of variance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "explained_variance.png", dpi=150)
    plt.close(fig)
    return X_tr, X_te, pca


# ---------------------------------------------------------------------------
# 3. TRAIN MODELS
# ---------------------------------------------------------------------------
def train_models(task, X_train, y_train):
    xgb_metric = "logloss" if task == "binary" else "mlogloss"
    models = {
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric=xgb_metric, n_jobs=-1),
    }
    timings = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        timings[name] = time.time() - t0
        print(f"  Trained {name} in {timings[name]:.1f}s")

    # Soft voting: averages predicted class probabilities of the three models
    t0 = time.time()
    voting = VotingClassifier(
        estimators=[("dt", models["decision_tree"]),
                    ("rf", models["random_forest"]),
                    ("xgb", models["xgboost"])],
        voting="soft", n_jobs=-1)
    voting.fit(X_train, y_train)
    timings["voting_ensemble"] = time.time() - t0
    models["voting_ensemble"] = voting
    print(f"  Trained voting_ensemble in {timings['voting_ensemble']:.1f}s")
    return models, timings


# ---------------------------------------------------------------------------
# 4. EVALUATE
# ---------------------------------------------------------------------------
def evaluate_binary(models, X_test, y_test, timings, out_dir):
    rows = []
    print(f"\n{'Model':<18}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}")
    print("-" * 56)
    for name, model in models.items():
        t0 = time.time()
        p = model.predict(X_test)
        pred_time = time.time() - t0     # wall-clock to predict the whole test set
        r = dict(model=name, accuracy=accuracy_score(y_test, p),
                 precision=precision_score(y_test, p), recall=recall_score(y_test, p),
                 f1=f1_score(y_test, p), train_time_s=timings[name], predict_time_s=pred_time)
        rows.append(r)
        print(f"{name:<18}{r['accuracy']:<10.4f}{r['precision']:<11.4f}{r['recall']:<9.4f}{r['f1']:<8.4f}")
        joblib.dump(model, out_dir / f"{name}.pkl")
    return pd.DataFrame(rows)


def evaluate_multiclass(models, X_test, y_test, timings, out_dir, class_names):
    rows = []
    print(f"\n{'Model':<18}{'Accuracy':<10}{'Macro-P':<9}{'Macro-R':<9}{'Macro-F1':<10}{'Wtd-F1':<8}")
    print("-" * 64)
    preds = {}
    for name, model in models.items():
        t0 = time.time()
        p = model.predict(X_test)
        pred_time = time.time() - t0     # wall-clock to predict the whole test set
        preds[name] = p
        r = dict(model=name, accuracy=accuracy_score(y_test, p),
                 macro_precision=precision_score(y_test, p, average="macro", zero_division=0),
                 macro_recall=recall_score(y_test, p, average="macro", zero_division=0),
                 macro_f1=f1_score(y_test, p, average="macro", zero_division=0),
                 weighted_f1=f1_score(y_test, p, average="weighted", zero_division=0),
                 train_time_s=timings[name], predict_time_s=pred_time)
        rows.append(r)
        print(f"{name:<18}{r['accuracy']:<10.4f}{r['macro_precision']:<9.4f}{r['macro_recall']:<9.4f}"
              f"{r['macro_f1']:<10.4f}{r['weighted_f1']:<8.4f}")
        joblib.dump(model, out_dir / f"{name}.pkl")

        report = classification_report(y_test, p, target_names=class_names,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(out_dir / f"per_class_{name}.csv")

    results = pd.DataFrame(rows)
    best = results.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    print(f"\nBest model by macro-F1: {best}")
    print(f"\n=== {best}: per-class report ===")
    print(classification_report(y_test, preds[best], target_names=class_names, zero_division=0))
    save_confusion_matrix(y_test, preds[best], class_names, best, out_dir)
    return results


def save_confusion_matrix(y_true, y_pred, class_names, model_name, out_dir):
    """Row-normalised 10x10 heatmap: each row shows where actual class X got predicted."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = cm_norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Multi-class confusion matrix (row-normalised) - {model_name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = out_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot -> {path}")


def compare_pca(tasks_run):
    """Print with/without-PCA comparison for task pairs trained in THIS run."""
    for base, metric in (("binary", "f1"), ("multiclass", "macro_f1")):
        if base in tasks_run and f"{base}_pca" in tasks_run:
            a = pd.read_csv(ARTIFACTS_DIR / base / "results_summary.csv").set_index("model")
            b = pd.read_csv(ARTIFACTS_DIR / f"{base}_pca" / "results_summary.csv").set_index("model")
            print("\n" + "=" * 64 + f"\nPCA EFFECT ({base}: {metric}, predict time)\n" + "=" * 64)
            print(f"{'Model':<18}{metric+' raw':<14}{'PCA':<9}{'predict s raw':<15}{'PCA':<8}")
            for m in a.index:
                print(f"{m:<18}{a.loc[m, metric]:<14.4f}{b.loc[m, metric]:<9.4f}"
                      f"{a.loc[m, 'predict_time_s']:<15.2f}{b.loc[m, 'predict_time_s']:<8.2f}")


def compare_smote(class_names):
    """Print before/after-SMOTE comparison if both result sets exist."""
    a, b = ARTIFACTS_DIR / "multiclass", ARTIFACTS_DIR / "multiclass_smote"
    if not ((a / "results_summary.csv").exists() and (b / "results_summary.csv").exists()):
        return
    ra = pd.read_csv(a / "results_summary.csv").set_index("model")
    rb = pd.read_csv(b / "results_summary.csv").set_index("model")
    print("\n" + "=" * 64 + "\nSMOTE EFFECT (macro-F1 / accuracy)\n" + "=" * 64)
    print(f"{'Model':<18}{'MacroF1 before':<16}{'after':<9}{'Acc before':<12}{'after':<8}")
    for m in ra.index:
        print(f"{m:<18}{ra.loc[m,'macro_f1']:<16.4f}{rb.loc[m,'macro_f1']:<9.4f}"
              f"{ra.loc[m,'accuracy']:<12.4f}{rb.loc[m,'accuracy']:<8.4f}")
    best = ra["macro_f1"].idxmax()
    pa = pd.read_csv(a / f"per_class_{best}.csv", index_col=0)
    pb = pd.read_csv(b / f"per_class_{best}.csv", index_col=0)
    print(f"\nPer-class F1, {best}: before -> after SMOTE")
    for c in class_names:
        print(f"  {c:<15}{pa.loc[c,'f1-score']:.3f} -> {pb.loc[c,'f1-score']:.3f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["binary", "multiclass", "multiclass_smote", "binary_pca",
                             "multiclass_pca", "both", "all"])
    args = ap.parse_args()
    tasks = {"both": ["binary", "multiclass"],
             "all": ["binary", "multiclass", "multiclass_smote", "binary_pca", "multiclass_pca"]
             }.get(args.task, [args.task])

    print("=" * 64 + "\nLOAD + PREPROCESS\n" + "=" * 64)
    train_df, test_df = load_data()
    X_train, X_test, targets, feature_names, label_encoder, train_x, scaler = preprocess(train_df, test_df)
    class_names = list(label_encoder.classes_)
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")
    print(f"\nFeatures: {X_train.shape[1]} | Classes (multiclass): {class_names}")
    X_train_pca = X_test_pca = None
    if any(t.endswith("_pca") for t in tasks):
        X_train_pca, X_test_pca, _ = fit_pca(X_train, X_test, feature_names)

    for task in tasks:
        print("\n" + "=" * 64 + f"\nTASK: {task.upper()}\n" + "=" * 64)
        out_dir = ARTIFACTS_DIR / task
        out_dir.mkdir(parents=True, exist_ok=True)
        kind = "multiclass" if task.startswith("multiclass") else "binary"
        y_train, y_test = targets[kind]
        X_fit, X_eval = X_train, X_test
        if task.endswith("_pca"):
            X_fit, X_eval = X_train_pca, X_test_pca
        if task == "multiclass_smote":
            X_fit, y_train = apply_smote(train_x, y_train, scaler, class_names)
        models, timings = train_models(kind, X_fit, y_train)
        if kind == "binary":
            results = evaluate_binary(models, X_eval, y_test, timings, out_dir)
        else:
            results = evaluate_multiclass(models, X_eval, y_test, timings, out_dir, class_names)
        results.to_csv(out_dir / "results_summary.csv", index=False)

    if "multiclass" in tasks and "multiclass_smote" in tasks:   # only compare results from THIS run
        compare_smote(class_names)
    compare_pca(tasks)
    print(f"\nDone. Artifacts saved to '{ARTIFACTS_DIR}/'. Run 'python demo.py' to try predictions.")


if __name__ == "__main__":
    main()
