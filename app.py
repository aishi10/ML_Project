"""
Cyber Attack Detection - Web App (Streamlit)
=============================================
Browser front end for the trained ensemble models. It reuses the loading and
preprocessing code from demo.py, so predictions match the command-line demo.

Run (after `python train_pipeline.py`):
    streamlit run app.py

Tabs:
    Try a sample   pick a held-out UNSW-NB15 test row (or a random one) and see the prediction
    Batch upload   upload a CSV of raw UNSW-NB15 rows and get predictions for all of them
    Model results  result tables and plots saved by train_pipeline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from demo import (KEY_FEATURES, MODEL_CHOICES, MODEL_LABELS, ARTIFACTS,
                  best_model_name, load_everything)

TASK_LABELS = {
    "binary": "Binary: Normal vs Attack",
    "multiclass": "Multi-class: 10 classes",
    "multiclass_smote": "Multi-class + SMOTE",
    "binary_pca": "Binary + PCA",
    "multiclass_pca": "Multi-class + PCA",
}
BINARY_NAMES = ["Normal", "Attack"]

st.set_page_config(page_title="Cyber Attack Detection", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trained_tasks():
    """Tasks that have at least one saved model."""
    return [t for t in TASK_LABELS
            if any((ARTIFACTS / t / f"{m}.pkl").exists() for m in MODEL_CHOICES)]


@st.cache_resource(show_spinner="Loading model...")
def get_context(task, model_name):
    return load_everything(task, model_name)


def encode_and_scale(df_raw, encoders, scaler, feature_names):
    """Same encoding + scaling as training, for many rows at once."""
    df = df_raw.copy()
    for col, le in encoders.items():
        known = set(le.classes_)
        df[col] = df[col].where(df[col].isin(known), le.classes_[0])   # unseen category -> first class
        df[col] = le.transform(df[col])
    return scaler.transform(df[feature_names].astype(float))


def class_names_for(task, label_encoder):
    return BINARY_NAMES if task.startswith("binary") else list(label_encoder.classes_)


def predict_frame(task, model, scaler, encoders, feature_names, label_encoder, df_raw):
    X = encode_and_scale(df_raw, encoders, scaler, feature_names)
    names = class_names_for(task, label_encoder)
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    return pred, proba, names


def actual_label(task, row, names):
    """Return the true class name for a row if the column exists, else None."""
    if task.startswith("binary"):
        return names[int(row["label"])] if "label" in row else None
    return row["attack_cat"] if "attack_cat" in row else None


# ---------------------------------------------------------------------------
# Sidebar: choose task and model
# ---------------------------------------------------------------------------
st.title("🛡️ Cyber Attack Detection Using Ensemble Learning")
st.caption("UNSW-NB15 · Decision Tree vs Random Forest vs XGBoost vs soft-voting ensemble")

tasks = trained_tasks()
if not tasks:
    st.error("No trained models found. Train first with `python train_pipeline.py`, then reload this page.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    task = st.selectbox("Task", tasks, format_func=lambda t: TASK_LABELS[t])
    available = [m for m in MODEL_CHOICES if (ARTIFACTS / task / f"{m}.pkl").exists()]
    default = best_model_name(task)
    model_name = st.selectbox("Model", available,
                              index=available.index(default) if default in available else 0,
                              format_func=lambda m: MODEL_LABELS[m])
    st.markdown("---")
    st.caption("Default model = best for the task by F1 (binary) or macro-F1 (multi-class). "
               "Samples come from the official UNSW-NB15 test set, which the models never saw in training.")

model, scaler, encoders, feature_names, label_encoder, X_test_raw = get_context(task, model_name)
tab_sample, tab_batch, tab_results = st.tabs(["Try a sample", "Batch upload", "Model results"])


# ---------------------------------------------------------------------------
# Tab 1: single held-out sample
# ---------------------------------------------------------------------------
with tab_sample:
    n = len(X_test_raw)
    if "row_idx" not in st.session_state:
        st.session_state.row_idx = 0
    c1, c2 = st.columns([3, 1])
    with c2:
        st.write("")
        if st.button("🎲 Random sample"):
            st.session_state.row_idx = int(np.random.randint(0, n))
    with c1:
        idx = st.number_input(f"Test row (0 to {n - 1})", min_value=0, max_value=n - 1,
                              key="row_idx", step=1)

    row = X_test_raw.iloc[int(idx)]
    pred, proba, names = predict_frame(task, model, scaler, encoders, feature_names,
                                       label_encoder, row.to_frame().T)
    predicted, conf = names[int(pred[0])], float(proba[0].max())
    actual = actual_label(task, row, names)

    left, right = st.columns(2)
    with left:
        st.subheader("Traffic features")
        key = [c for c in KEY_FEATURES if c in row.index]
        st.dataframe(pd.DataFrame({"feature": key, "value": [str(row[c]) for c in key]}),
                     hide_index=True)
        with st.expander("All raw features"):
            st.dataframe(row.astype(str).rename("value"))
    with right:
        st.subheader("Prediction")
        is_normal = predicted.lower() == "normal"
        (st.success if is_normal else st.error)(f"**{predicted.upper()}**  ·  confidence {conf * 100:.1f}%")
        if actual is not None:
            ok = predicted == actual
            (st.success if ok else st.warning)(f"Actual: **{str(actual).upper()}**  →  "
                                              f"{'correct ✅' if ok else 'wrong ❌'}")
        st.write("Class probabilities")
        st.bar_chart(pd.Series(proba[0], index=names, name="probability"))


# ---------------------------------------------------------------------------
# Tab 2: batch upload
# ---------------------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with raw UNSW-NB15 columns (same format as `artifacts/X_test_raw_sample.csv`). "
             "If it has a `label` / `attack_cat` column, accuracy is reported too.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df_up = pd.read_csv(up)
        missing = [c for c in feature_names if c not in df_up.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing[:10])}"
                     f"{' ...' if len(missing) > 10 else ''}")
        else:
            pred, proba, names = predict_frame(task, model, scaler, encoders, feature_names,
                                               label_encoder, df_up)
            out = pd.DataFrame({"prediction": [names[i] for i in pred],
                                "confidence_%": (proba.max(axis=1) * 100).round(1)})
            truth_col = "label" if task.startswith("binary") else "attack_cat"
            if truth_col in df_up.columns:
                truth = (df_up[truth_col].map(lambda v: names[int(v)]) if task.startswith("binary")
                         else df_up[truth_col])
                out["actual"] = truth.values
                acc = (out["prediction"] == out["actual"]).mean()
                st.metric("Accuracy on uploaded rows", f"{acc * 100:.2f}%")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("Predicted class counts")
                st.bar_chart(out["prediction"].value_counts())
            with c2:
                st.dataframe(out, height=320)
            st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                               file_name=f"predictions_{task}.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Tab 3: saved results
# ---------------------------------------------------------------------------
with tab_results:
    st.subheader(f"Results: {TASK_LABELS[task]}")
    res_path = ARTIFACTS / task / "results_summary.csv"
    if res_path.exists():
        res = pd.read_csv(res_path)
        st.dataframe(res.style.format({c: "{:.4f}" for c in res.columns if c != "model"}),
                     hide_index=True)
    cms = sorted((ARTIFACTS / task).glob("confusion_matrix_*.png"))
    if cms:
        st.image(str(cms[0]), caption=cms[0].stem.replace("_", " "))
    per_class = ARTIFACTS / task / f"per_class_{model_name}.csv"
    if per_class.exists():
        st.write(f"Per-class report: {MODEL_LABELS[model_name]}")
        st.dataframe(pd.read_csv(per_class, index_col=0).round(3))
    if task.endswith("_pca"):
        pca_png = ARTIFACTS / "pca" / "explained_variance.png"
        if pca_png.exists():
            st.image(str(pca_png), caption="PCA explained variance")
