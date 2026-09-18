import hashlib
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from data_io import load_recording, extract_window_features
from evaluate_events import compute_event_f1, extract_events_vectorized

st.set_page_config(layout="wide")
st.title("Sleep Apnea Event Detection — Team 14")

VAL_SUBJECTS_PATH = Path("data/val_subjects.csv")
VAL_PREDICTIONS_PATH = Path("data/val_predictions_tuned.csv")
legacy_tabs_available = VAL_SUBJECTS_PATH.exists() and VAL_PREDICTIONS_PATH.exists()
if legacy_tabs_available:
    val_subjects = pd.read_csv(VAL_SUBJECTS_PATH)
    predictions = pd.read_csv(VAL_PREDICTIONS_PATH)

# --- Upload & Inspect tab: live inference over a user-uploaded PATS .mat file, using
# the same LightGBM window classifier (data_io.py -> train_model.py) that produced the
# Waveform Explorer / AHI Accuracy tabs above. train_model.py never persisted the fitted
# model, so it was reproduced here: same subject split (verified to match val_subjects.csv
# exactly), same 30s/15s windowing, same feature set, same LGBMClassifier hyperparameters,
# same tuned per-class probability thresholds - saved to LEGACY_MODEL_PATH.
LEGACY_MODEL_PATH = Path("data/models/lightgbm_window_classifier.joblib")
LEGACY_DISPLAY_CHANNELS = ["airFlow", "SpO2", "thorax"]


@st.cache_resource
def load_legacy_model():
    return joblib.load(LEGACY_MODEL_PATH)


@st.cache_data(show_spinner=False)
def score_uploaded_file_legacy(file_bytes: bytes, _artifact: dict) -> dict:
    """Runs window-feature extraction + the LightGBM classifier once per uploaded
    file, cached on the file's own bytes so dragging the threshold sliders below
    doesn't re-trigger it. `_artifact` is excluded from the cache key (leading
    underscore) since it's just the fixed model this tab always uses.
    """
    digest = hashlib.sha1(file_bytes).hexdigest()[:16]
    tmp_path = Path(tempfile.gettempdir()) / f"sdb_dashboard_legacy_upload_{digest}.mat"
    if not tmp_path.exists():
        tmp_path.write_bytes(file_bytes)

    rec = load_recording(str(tmp_path))
    windows_df = extract_window_features(
        rec, window_seconds=_artifact["window_seconds"], step_seconds=_artifact["step_seconds"]
    )

    probabilities = _artifact["model"].predict_proba(windows_df[_artifact["feature_cols"]])
    classes = _artifact["classes"]
    windows_df["prob_apnea"] = probabilities[:, classes.index("apnea")]
    windows_df["prob_hypopnea"] = probabilities[:, classes.index("hypopnea")]
    windows_df["prob_normal"] = probabilities[:, classes.index("normal")]

    return {
        "rec": rec,
        "windows_df": windows_df,
        "true_ahi": float(rec.metrics["ahi"]),
        "recording_hours": float(rec.time[-1] / 3600.0),
    }


def apply_thresholds_legacy(windows_df: pd.DataFrame, apnea_threshold: float, hypop_threshold: float) -> pd.DataFrame:
    """Cheap, threshold-dependent step - reruns instantly as sliders move. Same tuned
    priority-order rule as train_model.py's __main__: apnea threshold checked first.
    """
    windows_df = windows_df.copy()
    windows_df["predicted_label"] = np.where(
        windows_df["prob_apnea"] > apnea_threshold, "apnea",
        np.where(windows_df["prob_hypopnea"] > hypop_threshold, "hypopnea", "normal"),
    )
    return windows_df


# true=cool colors, predicted=warm colors, apnea=blue/red, hypopnea=teal/orange - chosen so a
# correct detection (true+predicted overlapping, translucent) blends into a visibly distinct
# third color (purple for apnea, olive/brown for hypopnea) rather than disappearing, so overlap
# itself reads as "the model agrees with ground truth here" instead of being a rendering artifact.
EVENT_SPAN_STYLE = {
    "True apnea": "blue",
    "Predicted apnea": "red",
    "True hypopnea": "teal",
    "Predicted hypopnea": "orange",
}


def render_legacy_waveform_and_confidence(rec, windows_df: pd.DataFrame):
    true_apneas = extract_events_vectorized(windows_df.rename(columns={"predicted_label": "_", "label": "predicted_label"}), "apnea")
    pred_apneas = extract_events_vectorized(windows_df, "apnea")
    true_hypops = extract_events_vectorized(windows_df.rename(columns={"predicted_label": "_", "label": "predicted_label"}), "hypopnea")
    pred_hypops = extract_events_vectorized(windows_df, "hypopnea")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.05,
        subplot_titles=(
            "Waveforms (z-scored) — overlapping true/predicted shading blends into a third "
            "color where the model agrees with ground truth (see legend)",
            "Model confidence (per-window class probability)",
        ),
    )

    for i, channel in enumerate(LEGACY_DISPLAY_CHANNELS):
        idx = rec.labels.index(channel)
        raw = rec.data[:, idx]
        normalized = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)
        fig.add_trace(go.Scatter(x=rec.time, y=normalized + i * 6, name=channel, line=dict(width=1)), row=1, col=1)

    # legend-only entries - vrect shapes can't carry a legend themselves, so a zero-size
    # marker trace per span type stands in for one, making the true/predicted color mapping
    # explicit instead of relying on the caption alone.
    for label, color in EVENT_SPAN_STYLE.items():
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, symbol="square", color=color), name=label),
            row=1, col=1,
        )

    for start, end in true_apneas:
        fig.add_vrect(x0=rec.time[start], x1=rec.time[end], fillcolor=EVENT_SPAN_STYLE["True apnea"], opacity=0.25, line_width=0, row=1, col=1)
    for start, end in pred_apneas:
        fig.add_vrect(x0=rec.time[start], x1=rec.time[end], fillcolor=EVENT_SPAN_STYLE["Predicted apnea"], opacity=0.25, line_width=0, row=1, col=1)
    for start, end in true_hypops:
        fig.add_vrect(x0=rec.time[start], x1=rec.time[end], fillcolor=EVENT_SPAN_STYLE["True hypopnea"], opacity=0.25, line_width=0, row=1, col=1)
    for start, end in pred_hypops:
        fig.add_vrect(x0=rec.time[start], x1=rec.time[end], fillcolor=EVENT_SPAN_STYLE["Predicted hypopnea"], opacity=0.25, line_width=0, row=1, col=1)

    window_mid_idx = ((windows_df["start"] + windows_df["end"]) // 2).to_numpy()
    window_mid_time = rec.time[window_mid_idx]
    fig.add_trace(go.Scatter(x=window_mid_time, y=windows_df["prob_apnea"], name="P(apnea)", line=dict(width=1, color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=window_mid_time, y=windows_df["prob_hypopnea"], name="P(hypopnea)", line=dict(width=1, color="orange")), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=750, legend=dict(orientation="h", y=1.08))
    return fig


tab1, tab2, tab3 = st.tabs(["Waveform Explorer", "AHI Accuracy", "Upload & Inspect"])

with tab1:
    if not legacy_tabs_available:
        st.info(
            f"This tab needs {VAL_SUBJECTS_PATH} and {VAL_PREDICTIONS_PATH}, which aren't "
            "in the repo yet - use the Upload & Inspect tab instead."
        )
    else:
        subject_ids = sorted(val_subjects["study_number"].unique())
        selected = st.selectbox("Select a subject", subject_ids)

        filepath = f"data/ahiData-pats-{selected}-baseline.mat"
        rec = load_recording(filepath)

        subject_preds = predictions[predictions["study_number"] == selected].sort_values("start")
        true_apneas = extract_events_vectorized(subject_preds.rename(columns={"predicted_label": "_", "label": "predicted_label"}), "apnea")
        pred_apneas = extract_events_vectorized(subject_preds, "apnea")
        true_hypops = extract_events_vectorized(subject_preds.rename(columns={"predicted_label": "_", "label": "predicted_label"}), "hypopnea")
        pred_hypops = extract_events_vectorized(subject_preds, "hypopnea")

        channels_to_show = ["airFlow", "SpO2", "thorax"]
        fig = go.Figure()

        for i, ch in enumerate(channels_to_show):
            idx = rec.labels.index(ch)
            raw = rec.data[:, idx]
            normalized = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)  # z-score, avoids divide-by-zero
            fig.add_trace(go.Scatter(x=rec.time, y=normalized + i * 6, name=ch, line=dict(width=1)))

        for start, end in true_apneas:
            fig.add_vrect(x0=rec.time[start], x1=rec.time[end], fillcolor="blue", opacity=0.2, line_width=0)
        for start, end in pred_apneas:
            fig.add_vrect(x0=rec.time[start], x1=rec.time[end], fillcolor="red", opacity=0.2, line_width=0)

        fig.update_layout(height=600, title=f"Subject {selected} — blue=true apnea, red=predicted apnea")
        st.plotly_chart(fig, width="stretch")

with tab2:
    if not legacy_tabs_available:
        st.info(
            f"This tab needs {VAL_SUBJECTS_PATH} and {VAL_PREDICTIONS_PATH}, which aren't "
            "in the repo yet - use the Upload & Inspect tab instead."
        )
    else:
        st.subheader("Predicted vs. True AHI (Validation Set)")

        ahi_data = []
        for study_number in val_subjects["study_number"]:
            true_ahi = val_subjects.loc[val_subjects["study_number"] == study_number, "ahi"].values[0]
            subject_preds = predictions[predictions["study_number"] == study_number].sort_values("start")
            apneas = extract_events_vectorized(subject_preds, "apnea")
            hypops = extract_events_vectorized(subject_preds, "hypopnea")
            hours = val_subjects.loc[val_subjects["study_number"] == study_number, "recording_hours"].values[0]
            pred_ahi = (len(apneas) + len(hypops)) / hours
            ahi_data.append({"study_number": study_number, "true_ahi": true_ahi, "predicted_ahi": pred_ahi})

        ahi_df = pd.DataFrame(ahi_data)
        max_val = max(ahi_df["true_ahi"].max(), ahi_df["predicted_ahi"].max())

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ahi_df["true_ahi"], y=ahi_df["predicted_ahi"], mode="markers", marker=dict(size=10)))
        fig2.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect prediction"))
        fig2.update_layout(xaxis_title="True AHI", yaxis_title="Predicted AHI", height=600)
        st.plotly_chart(fig2, width="stretch")

        mae = np.mean(np.abs(ahi_df["true_ahi"] - ahi_df["predicted_ahi"]))
        st.metric("AHI Mean Absolute Error", f"{mae:.2f} events/hour")

with tab3:
    st.subheader("Upload a recording and run live inference")
    st.caption(
        f"Runs `{LEGACY_MODEL_PATH.name}` — the same LightGBM window classifier (30s "
        "windows / 15s stride, per-channel mean/std/min/max/range/slope features, tuned "
        "per-class probability thresholds) trained on the same subject split behind the "
        "Waveform Explorer and AHI Accuracy tabs above — over the uploaded recording."
    )

    uploaded = st.file_uploader("Upload a PATS baseline .mat file", type=["mat"])

    if uploaded is None:
        st.info("Upload a PATS baseline .mat file (schema: signals/events/metrics structs) to run live inference.")
    else:
        try:
            artifact = load_legacy_model()
        except Exception as error:  # noqa: BLE001 - surface as a dashboard message, not a crash
            st.error(f"Could not load model at {LEGACY_MODEL_PATH}: {error}")
            st.stop()

        try:
            with st.spinner("Extracting window features and scoring..."):
                result = score_uploaded_file_legacy(uploaded.getvalue(), artifact)
        except Exception as error:  # noqa: BLE001 - degrade gracefully on unexpected/malformed files, per CLAUDE.md
            st.error(
                "Couldn't process this file - it may not match the expected PATS .mat schema "
                f"(signals/events/metrics structs). Details: {error}"
            )
            st.stop()

        st.markdown("**Detection thresholds** (recompute instantly - no need to re-upload)")
        col1, col2 = st.columns(2)
        apnea_threshold = col1.slider(
            "Apnea probability threshold", 0.0, 1.0, artifact["best_apnea_threshold"], 0.05,
        )
        hypop_threshold = col2.slider(
            "Hypopnea probability threshold", 0.0, 1.0, artifact["best_hypop_threshold"], 0.05,
        )

        windows_df = apply_thresholds_legacy(result["windows_df"], apnea_threshold, hypop_threshold)
        fig3 = render_legacy_waveform_and_confidence(result["rec"], windows_df)

        apnea_p, apnea_r, apnea_f1 = compute_event_f1(windows_df, "apnea")
        hypop_p, hypop_r, hypop_f1 = compute_event_f1(windows_df, "hypopnea")
        pred_apneas = extract_events_vectorized(windows_df, "apnea")
        pred_hypops = extract_events_vectorized(windows_df, "hypopnea")
        pred_ahi = (len(pred_apneas) + len(pred_hypops)) / result["recording_hours"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted AHI", f"{pred_ahi:.1f}")
        m2.metric(
            "True AHI", f"{result['true_ahi']:.1f}",
            delta=f"{pred_ahi - result['true_ahi']:+.1f}", delta_color="inverse",
        )
        m3.metric("Apnea F1", f"{apnea_f1:.2f}", help=f"precision={apnea_p:.2f} recall={apnea_r:.2f}")
        m4.metric("Hypopnea F1", f"{hypop_f1:.2f}", help=f"precision={hypop_p:.2f} recall={hypop_r:.2f}")

        st.plotly_chart(fig3, width="stretch")