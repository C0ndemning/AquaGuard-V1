from __future__ import annotations

import io
import json
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from feature_pipeline import FLOW_COLUMNS, prepare_model_matrix

st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "deployment_model.pkl"
FEATURES_PATH = MODELS / "deployment_features.pkl"
ENCODER_PATH = MODELS / "deployment_label_encoder.pkl"
METADATA_PATH = MODELS / "deployment_metadata.json"

PRESSURE_COLUMNS = [
    "n1", "n4", "n31", "n54", "n105", "n114", "n163", "n188",
    "n215", "n229", "n288", "n296", "n332", "n342", "n410",
    "n415", "n429", "n458", "n469", "n495", "n506", "n516",
    "n519", "n549", "n613", "n636", "n644", "n679", "n722",
    "n726", "n740", "n752", "n769",
]
RAW_COLUMNS = PRESSURE_COLUMNS + FLOW_COLUMNS
MIN_HISTORY = 6

st.markdown(
    """
    <style>
    :root { --panel:#0d1c2b; --line:#244159; --muted:#8fa9bd; }
    .stApp {
        background:
          radial-gradient(circle at 14% 0%, rgba(22,95,150,.18), transparent 30%),
          linear-gradient(180deg,#06101a 0%,#091522 100%);
    }
    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#091827,#06111c);
        border-right:1px solid #20384b;
    }
    .block-container {max-width:1800px; padding-top:1.05rem;}
    .hero {
        padding:18px 22px; border:1px solid #25445d; border-radius:18px;
        background:linear-gradient(135deg,rgba(15,43,65,.96),rgba(8,25,40,.96));
        box-shadow:0 16px 40px rgba(0,0,0,.24); margin-bottom:14px;
    }
    .hero-title {font-size:1.62rem;font-weight:850;letter-spacing:.2px}
    .hero-sub {color:#93aec2;margin-top:3px}
    .chip {
        display:inline-block;padding:5px 10px;border-radius:999px;
        font-size:.74rem;font-weight:800;margin-right:7px;margin-top:9px;
    }
    .green {color:#6ce6ad;border:1px solid #2d9e6b;background:rgba(39,190,119,.13)}
    .amber {color:#ffd477;border:1px solid #a47b26;background:rgba(244,184,55,.13)}
    .red {color:#ff8f8f;border:1px solid #a44242;background:rgba(238,77,77,.13)}
    .blue {color:#86c3ff;border:1px solid #376fa6;background:rgba(49,133,215,.13)}
    [data-testid="stMetric"] {
        background:linear-gradient(180deg,rgba(14,39,59,.97),rgba(8,25,40,.97));
        border:1px solid #25445d;border-radius:14px;padding:12px 14px;
    }
    .panel {
        border:1px solid #25445d;border-radius:15px;padding:15px;
        background:linear-gradient(180deg,rgba(12,32,50,.97),rgba(8,23,37,.97));
    }
    .alarm {padding:10px 12px;border-left:4px solid;border-radius:9px;margin:6px 0;background:rgba(255,255,255,.035)}
    .alarm-critical {border-color:#f45b5b}.alarm-warning{border-color:#efb83f}.alarm-info{border-color:#3e95d4}
    .muted {color:#8fa9bd;font-size:.84rem}
    .stTabs [data-baseweb="tab"] {background:#0d2336;border:1px solid #244159;border-radius:9px;padding:8px 14px}
    .stTabs [aria-selected="true"] {background:#174a70}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    required = [MODEL_PATH, FEATURES_PATH, ENCODER_PATH, METADATA_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing deployment artifacts: " + ", ".join(missing))

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    if len(features) != int(metadata["feature_count"]):
        raise ValueError("Deployment feature count does not match metadata.")
    if getattr(model, "n_features_in_", len(features)) != len(features):
        raise ValueError("Model input count does not match deployment feature list.")

    return model, list(features), encoder, metadata


def synthetic_history(rows: int = 72, seed: int = 42) -> pd.DataFrame:
    """Create a stable demonstration stream. It is not labelled test data."""
    rng = np.random.default_rng(seed)
    time = pd.date_range(end=pd.Timestamp.now().floor("5min"), periods=rows, freq="5min")
    phase = np.linspace(0, 3 * np.pi, rows)
    data: dict[str, object] = {"Timestamp": time}

    for index, sensor in enumerate(PRESSURE_COLUMNS):
        position_drop = index * 0.20
        base = 51.5 - position_drop
        demand_wave = 1.2 * np.sin(phase + index * 0.05)
        data[sensor] = base + demand_wave + rng.normal(0, 0.12, rows)

    data["p227"] = 22.0 + 2.3 * np.sin(phase) + rng.normal(0, 0.18, rows)
    data["p235"] = 18.0 + 1.9 * np.sin(phase + 0.35) + rng.normal(0, 0.16, rows)
    data["PUMP_1"] = 42.0 + 3.4 * np.sin(phase + 0.15) + rng.normal(0, 0.20, rows)
    return pd.DataFrame(data)


def validate_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"CSV is missing {len(missing)} required sensor columns: "
            + ", ".join(missing)
        )
    if "Timestamp" not in frame.columns:
        frame.insert(
            0,
            "Timestamp",
            pd.date_range(
                end=pd.Timestamp.now().floor("5min"),
                periods=len(frame),
                freq="5min",
            ),
        )
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    if frame["Timestamp"].isna().any():
        raise ValueError("One or more Timestamp values could not be parsed.")
    frame = frame.sort_values("Timestamp").reset_index(drop=True)
    if len(frame) < MIN_HISTORY:
        raise ValueError(
            f"At least {MIN_HISTORY} chronological rows are required because "
            "the model uses rolling and change features."
        )
    return frame


def apply_scenario(
    frame: pd.DataFrame,
    leak_strength: float,
    leak_center: str,
    noise_percent: float,
    missing_percent: float,
    seed: int,
) -> pd.DataFrame:
    output = frame.copy()
    rng = np.random.default_rng(seed)

    if leak_strength > 0:
        center = PRESSURE_COLUMNS.index(leak_center)
        tail = max(6, min(24, len(output)))
        ramp = np.linspace(0.15, 1.0, tail)
        for index, sensor in enumerate(PRESSURE_COLUMNS):
            distance = abs(index - center)
            effect = leak_strength * np.exp(-distance / 5.5)
            output.loc[output.index[-tail:], sensor] -= effect * ramp
        output.loc[output.index[-tail:], "PUMP_1"] += leak_strength * 0.55 * ramp
        output.loc[output.index[-tail:], "p227"] += leak_strength * 0.18 * ramp

    if noise_percent > 0:
        for column in RAW_COLUMNS:
            scale = max(float(output[column].std()), 1e-6)
            output[column] += rng.normal(
                0.0, scale * noise_percent / 100.0, len(output)
            )

    if missing_percent > 0:
        for column in RAW_COLUMNS:
            count = int(round(len(output) * missing_percent / 100.0))
            if count:
                indices = rng.choice(output.index, size=min(count, len(output)), replace=False)
                output.loc[indices, column] = np.nan
        output[RAW_COLUMNS] = (
            output[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce").ffill().bfill()
        )

    return output


def infer(frame, model, features, encoder):
    matrix = prepare_model_matrix(frame, features)
    probabilities = model.predict_proba(matrix)
    encoded = np.argmax(probabilities, axis=1)
    labels = encoder.inverse_transform(encoded)
    confidence = probabilities.max(axis=1)
    result = pd.DataFrame(
        {
            "Timestamp": frame["Timestamp"].to_numpy(),
            "prediction": labels,
            "confidence": confidence,
        }
    )
    for index, class_name in enumerate(encoder.classes_):
        result[f"probability_{class_name}"] = probabilities[:, index]
    return result, matrix


def sensor_health(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in RAW_COLUMNS:
        series = pd.to_numeric(frame[column], errors="coerce")
        missing = float(series.isna().mean())
        changes = series.diff().abs()
        stuck = bool(len(series) >= 6 and changes.tail(5).fillna(0).max() < 1e-9)
        z = (series - series.mean()) / max(float(series.std()), 1e-9)
        spike_count = int((z.abs() > 4.0).sum())
        if missing > 0.10 or stuck:
            status = "Critical"
        elif missing > 0 or spike_count:
            status = "Warning"
        else:
            status = "Healthy"
        rows.append(
            {
                "sensor": column,
                "status": status,
                "missing_%": missing * 100,
                "stuck": stuck,
                "spikes": spike_count,
                "latest": float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def control_recommendation(label: str, confidence: float, threshold: float) -> tuple[str, str]:
    if confidence < threshold:
        return "HOLD", "Confidence is below the operating threshold. Require operator review."
    if label == "none":
        return "MONITOR", "Keep normal operation and continue sensor-health checks."
    return (
        "ISOLATE / VERIFY",
        f"Verify the area associated with {label}, inspect nearby sensors, "
        "and isolate only after hydraulic confirmation.",
    )


def probability_chart(row: pd.Series, classes: list[str]) -> go.Figure:
    values = [float(row[f"probability_{name}"]) for name in classes]
    order = np.argsort(values)[::-1][:8]
    labels = [classes[index] for index in order][::-1]
    values = [values[index] * 100 for index in order][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h"))
    fig.update_layout(
        title="Top class probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=390,
        margin=dict(l=20, r=20, t=55, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dcebf7"),
    )
    return fig


def pressure_chart(frame: pd.DataFrame, highlighted: str) -> go.Figure:
    latest = frame[PRESSURE_COLUMNS].iloc[-1]
    fig = go.Figure(
        go.Scatter(
            x=PRESSURE_COLUMNS,
            y=latest.values,
            mode="lines+markers",
            name="Latest pressure",
        )
    )
    if highlighted in PRESSURE_COLUMNS:
        fig.add_trace(
            go.Scatter(
                x=[highlighted],
                y=[latest[highlighted]],
                mode="markers",
                marker=dict(size=15, symbol="diamond"),
                name="Scenario center",
            )
        )
    fig.update_layout(
        title="Pressure profile across monitored nodes",
        yaxis_title="Sensor reading",
        xaxis_title="Pressure sensor",
        height=390,
        margin=dict(l=20, r=20, t=55, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dcebf7"),
    )
    return fig


try:
    model, saved_features, encoder, metadata = load_artifacts()
    artifact_error = None
except Exception as error:
    model = saved_features = encoder = metadata = None
    artifact_error = str(error)

if "raw_data" not in st.session_state:
    st.session_state.raw_data = synthetic_history()
if "source_name" not in st.session_state:
    st.session_state.source_name = "Demonstration stream"

with st.sidebar:
    st.markdown("## AquaGuard AI")
    page = st.radio(
        "Workspace",
        [
            "Control Room",
            "AI Decision Center",
            "Sensor Health",
            "Data & Scenarios",
            "Model Card",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    threshold = st.slider(
        "Operating confidence threshold",
        0.00,
        1.00,
        0.80,
        0.01,
    )

    st.markdown("#### Scenario controls")
    leak_strength = st.slider("Pressure disturbance", 0.0, 15.0, 0.0, 0.25)
    leak_center = st.selectbox("Disturbance center", PRESSURE_COLUMNS, index=26)
    noise_percent = st.slider("Sensor noise (%)", 0.0, 20.0, 0.0, 0.5)
    missing_percent = st.slider("Missing readings (%)", 0.0, 20.0, 0.0, 0.5)
    scenario_seed = st.number_input("Scenario seed", 1, 999999, 42)

    if st.button("Reset demonstration data", use_container_width=True):
        st.session_state.raw_data = synthetic_history(seed=int(scenario_seed))
        st.session_state.source_name = "Demonstration stream"
        st.rerun()

    st.caption(
        "Scenario controls alter raw sensor history. The displayed prediction "
        "still comes from the uploaded production XGBoost model."
    )

status_chip = (
    '<span class="chip green">MODEL ONLINE</span>'
    if artifact_error is None
    else '<span class="chip red">MODEL ERROR</span>'
)
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">AquaGuard AI — Leak Localization Command Center</div>
      <div class="hero-sub">
        Real inference using the selected 205-feature XGBoost deployment model.
      </div>
      {status_chip}
      <span class="chip blue">{st.session_state.source_name.upper()}</span>
      <span class="chip amber">OPERATOR-IN-THE-LOOP</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if artifact_error:
    st.error(artifact_error)
    st.stop()

try:
    raw = validate_raw_frame(st.session_state.raw_data)
    scenario = apply_scenario(
        raw,
        leak_strength=float(leak_strength),
        leak_center=leak_center,
        noise_percent=float(noise_percent),
        missing_percent=float(missing_percent),
        seed=int(scenario_seed),
    )
    predictions, model_matrix = infer(
        scenario, model, saved_features, encoder
    )
except Exception as error:
    st.error(f"Inference pipeline failed: {error}")
    st.stop()

latest = predictions.iloc[-1]
latest_label = str(latest["prediction"])
latest_confidence = float(latest["confidence"])
action, action_reason = control_recommendation(
    latest_label, latest_confidence, threshold
)
health = sensor_health(scenario)
healthy_count = int((health["status"] == "Healthy").sum())
warning_count = int((health["status"] == "Warning").sum())
critical_count = int((health["status"] == "Critical").sum())

if page == "Control Room":
    columns = st.columns(5)
    columns[0].metric("AI state", latest_label.upper())
    columns[1].metric("Confidence", f"{latest_confidence*100:.1f}%")
    columns[2].metric("Decision", action)
    columns[3].metric("Healthy sensors", f"{healthy_count}/{len(health)}")
    columns[4].metric("Feature vector", f"{model_matrix.shape[1]}")

    if latest_confidence < threshold:
        st.markdown(
            '<div class="alarm alarm-warning"><b>Prediction withheld:</b> '
            'confidence is below the selected operating threshold.</div>',
            unsafe_allow_html=True,
        )
    elif latest_label != "none":
        st.markdown(
            f'<div class="alarm alarm-critical"><b>Leak-location alert:</b> '
            f'the model predicts <b>{latest_label}</b> at '
            f'<b>{latest_confidence*100:.1f}%</b> confidence.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alarm alarm-info"><b>Normal-state prediction:</b> '
            'continue monitoring; this is not a guarantee that the physical '
            'system is fault-free.</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(
            pressure_chart(scenario, leak_center),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            probability_chart(latest, list(encoder.classes_)),
            use_container_width=True,
        )

    st.markdown("### Decision record")
    st.info(f"**Recommended state:** {action}\n\n{action_reason}")
    record = pd.DataFrame(
        {
            "time": predictions["Timestamp"].tail(20),
            "prediction": predictions["prediction"].tail(20),
            "confidence_%": predictions["confidence"].tail(20) * 100,
        }
    )
    st.dataframe(record, use_container_width=True, hide_index=True)

elif page == "AI Decision Center":
    st.subheader("AI output and uncertainty")
    top_classes = (
        pd.DataFrame(
            {
                "class": list(encoder.classes_),
                "probability": [
                    latest[f"probability_{name}"] for name in encoder.classes_
                ],
            }
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    top_classes["probability_%"] = top_classes["probability"] * 100

    left, right = st.columns([1, 1.25])
    with left:
        st.metric("Predicted class", latest_label)
        st.metric("Top probability", f"{latest_confidence*100:.2f}%")
        margin = (
            float(top_classes.loc[0, "probability"])
            - float(top_classes.loc[1, "probability"])
        )
        st.metric("Top-two probability margin", f"{margin*100:.2f} points")
        st.metric("Threshold", f"{threshold*100:.0f}%")
        st.info(f"**Controller recommendation:** {action}\n\n{action_reason}")
    with right:
        st.plotly_chart(
            probability_chart(latest, list(encoder.classes_)),
            use_container_width=True,
        )

    st.subheader("Complete probability output")
    st.dataframe(
        top_classes[["class", "probability_%"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The dashboard does not invent a natural-language model explanation. "
        "It reports real class probabilities and keeps control logic separate "
        "from the classifier."
    )

elif page == "Sensor Health":
    st.subheader("Raw-input quality gate")
    metrics = st.columns(4)
    metrics[0].metric("Healthy", healthy_count)
    metrics[1].metric("Warnings", warning_count)
    metrics[2].metric("Critical", critical_count)
    metrics[3].metric("Total sensors", len(health))

    health_filter = st.multiselect(
        "Show statuses",
        ["Healthy", "Warning", "Critical"],
        default=["Warning", "Critical"] if warning_count + critical_count else ["Healthy"],
    )
    displayed = health[health["status"].isin(health_filter)]
    st.dataframe(displayed, use_container_width=True, hide_index=True)

    st.markdown("### Input history")
    selected_sensor = st.selectbox("Sensor", RAW_COLUMNS)
    fig = go.Figure(
        go.Scatter(
            x=scenario["Timestamp"],
            y=scenario[selected_sensor],
            mode="lines",
            name=selected_sensor,
        )
    )
    fig.update_layout(
        height=380,
        yaxis_title="Reading",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dcebf7"),
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Data & Scenarios":
    st.subheader("Load a real sensor stream")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            candidate = pd.read_csv(uploaded)
            candidate = validate_raw_frame(candidate)
            st.success(
                f"Validated {len(candidate):,} chronological rows and "
                f"{len(RAW_COLUMNS)} required sensors."
            )
            if st.button("Use this CSV for inference", type="primary"):
                st.session_state.raw_data = candidate
                st.session_state.source_name = uploaded.name
                st.rerun()
        except Exception as error:
            st.error(str(error))

    template = synthetic_history(rows=12)
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="aquaguard_sensor_template.csv",
        mime="text/csv",
    )

    st.subheader("Current raw data")
    st.dataframe(scenario.tail(25), use_container_width=True, hide_index=True)
    st.caption(
        "For live deployment, append each new 5-minute sensor row to the "
        "history buffer and run the exact same feature pipeline."
    )

elif page == "Model Card":
    st.subheader("Production deployment model")
    card = {
        "Role": metadata["selection_role"],
        "Algorithm": metadata["model_name"],
        "Features": metadata["feature_count"],
        "Classes": len(encoder.classes_),
        "Accuracy": f"{metadata['accuracy']*100:.2f}%",
        "Balanced accuracy": f"{metadata['balanced_accuracy']*100:.2f}%",
        "Macro F1": f"{metadata['macro_f1']*100:.2f}%",
        "Weighted F1": f"{metadata['weighted_f1']*100:.2f}%",
        "Macro precision": f"{metadata['macro_precision']*100:.2f}%",
        "Macro recall": f"{metadata['macro_recall']*100:.2f}%",
        "ROC AUC, macro OVR": f"{metadata['roc_auc_macro_ovr']:.4f}",
        "PR AUC, macro": f"{metadata['pr_auc_macro']:.4f}",
        "Model size": f"{metadata['model_size_mb']:.2f} MB",
        "Inference": f"{metadata['prediction_ms_per_sample']:.4f} ms/sample",
    }
    st.dataframe(
        pd.DataFrame(card.items(), columns=["Field", "Value"]),
        use_container_width=True,
        hide_index=True,
    )
    st.info(metadata["selection_rule"])
    st.warning(
        "Benchmark performance is not proof of physical-model performance. "
        "Before autonomous control, validate with the real hydraulic system, "
        "real sensor calibration errors, unseen operating conditions, and "
        "fail-safe operator procedures."
    )
    st.markdown("#### Leak classes")
    st.code(", ".join(map(str, encoder.classes_)))
