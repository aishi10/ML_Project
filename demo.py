"""
Cyber Attack Detection Demo
============================
Loads the trained ensemble models and tests predictions on real network
traffic samples from the held-out UNSW-NB15 test set (never seen in training).

Usage:
    python demo.py                              # multi-class, interactive
    python demo.py --task binary                # Normal vs Attack
    python demo.py --task multiclass_smote      # multi-class models trained with SMOTE
    python demo.py --task binary_pca            # binary / multiclass_pca: models trained on PCA features
    python demo.py --random 5                   # 5 random predictions, then exit
    python demo.py --index 42                   # one specific test row
    python demo.py --model random_forest        # pick a model
                                                # (decision_tree / random_forest / xgboost / voting_ensemble)
Default model = best one for the chosen task (by F1 for binary, macro-F1 for multi-class).
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_CHOICES = ["decision_tree", "random_forest", "xgboost", "voting_ensemble"]
MODEL_LABELS = {
    "decision_tree": "Decision Tree (Baseline)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "voting_ensemble": "Voting Ensemble (DT + RF + XGBoost)",
}
KEY_FEATURES = ["dur", "proto", "service", "state", "sbytes", "dbytes", "sttl", "rate"]


def best_model_name(task):
    """Pick the top model for this task from the saved results table."""
    try:
        res = pd.read_csv(ARTIFACTS / task / "results_summary.csv")
        metric = "f1" if task.startswith("binary") else "macro_f1"
        return res.sort_values(metric, ascending=False).iloc[0]["model"]
    except Exception:
        return "xgboost"


class _ScalerThenPCA:
    """Lets PCA-task models reuse the same preprocessing call as the other tasks."""
    def __init__(self, scaler, pca):
        self.scaler, self.pca = scaler, pca

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))


def load_everything(task, model_name):
    model_path = ARTIFACTS / task / f"{model_name}.pkl"
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\n"
                         f"Train it first:  python train_pipeline.py --task {task}")
    model = joblib.load(model_path)
    scaler = joblib.load(ARTIFACTS / "scaler.pkl")
    if task.endswith("_pca"):
        scaler = _ScalerThenPCA(scaler, joblib.load(ARTIFACTS / "pca" / "pca.pkl"))
    encoders = joblib.load(ARTIFACTS / "encoders.pkl")
    feature_names = joblib.load(ARTIFACTS / "feature_names.pkl")
    label_encoder = joblib.load(ARTIFACTS / "label_encoder.pkl")
    X_test_raw = pd.read_csv(ARTIFACTS / "X_test_raw_sample.csv")
    return model, scaler, encoders, feature_names, label_encoder, X_test_raw


def preprocess_row(row_raw, encoders, scaler, feature_names):
    """Apply the same encoding + scaling used in training to one raw row."""
    row = row_raw.copy()
    for col, le in encoders.items():
        if row[col] not in le.classes_:      # safety net for unseen categories
            row[col] = le.classes_[0]
        row[col] = le.transform([row[col]])[0]
    row_df = row[feature_names].to_frame().T.astype(float)
    return scaler.transform(row_df)


def predict_one(task, model, scaler, encoders, feature_names, label_encoder, row_raw):
    X = preprocess_row(row_raw, encoders, scaler, feature_names)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    print("-" * 62)
    print("Key traffic features:")
    for col in KEY_FEATURES:
        if col in row_raw:
            print(f"    {col:<10} : {row_raw[col]}")
    print()

    if task.startswith("binary"):
        names = {0: "NORMAL", 1: "ATTACK"}
        actual = int(row_raw["label"])
        print(f"  PREDICTION : {names[pred]}")
        print(f"  Confidence : Normal={proba[0]*100:.1f}%  Attack={proba[1]*100:.1f}%")
        ok = pred == actual
        print(f"  Actual     : {names[actual]}   [{'CORRECT' if ok else 'WRONG'}]")
    else:
        classes = list(label_encoder.classes_)
        actual = row_raw["attack_cat"]
        predicted = classes[pred]
        print(f"  PREDICTION : {predicted.upper()}")
        print("  Top 3      : " + "   ".join(
            f"{classes[i]} {proba[i]*100:.1f}%" for i in np.argsort(proba)[::-1][:3]))
        ok = predicted == actual
        print(f"  Actual     : {actual.upper()}   [{'CORRECT' if ok else 'WRONG'}]")
    print("-" * 62)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Cyber Attack Detection - Demo")
    ap.add_argument("--task", choices=["binary", "multiclass", "multiclass_smote", "binary_pca", "multiclass_pca"],
                    default="multiclass")
    ap.add_argument("--model", choices=MODEL_CHOICES, default=None)
    ap.add_argument("--random", type=int, help="Show N random predictions and exit")
    ap.add_argument("--index", type=int, help="Predict one specific test-set row")
    args = ap.parse_args()

    model_name = args.model or best_model_name(args.task)
    print(f"\nTask: {args.task} | Model: {MODEL_LABELS[model_name]}\n")
    ctx = load_everything(args.task, model_name)
    model, scaler, encoders, feature_names, label_encoder, X_test_raw = ctx

    def run(i):
        return predict_one(args.task, model, scaler, encoders, feature_names,
                           label_encoder, X_test_raw.iloc[i])

    if args.index is not None:
        run(args.index)
        return

    if args.random:
        idxs = np.random.choice(len(X_test_raw), size=args.random, replace=False)
        correct = sum(run(i) for i in idxs)
        print(f"\n{correct}/{len(idxs)} correct")
        return

    print("Interactive demo.")
    while True:
        cmd = input("\n[Enter]=random sample, [number]=specific row, [q]=quit: ").strip()
        if cmd.lower() == "q":
            break
        if cmd == "":
            run(np.random.randint(0, len(X_test_raw)))
        elif cmd.isdigit() and int(cmd) < len(X_test_raw):
            run(int(cmd))
        else:
            print(f"Enter a number 0-{len(X_test_raw)-1}, press Enter, or q.")


if __name__ == "__main__":
    main()
