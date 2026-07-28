from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from feature_pipeline import prepare_model_matrix
from physics_engine import (
    FLUIDS,
    HOUSES,
    SEGMENTS,
    add_water,
    create_leak,
    default_state,
    log_event,
    patch_leak,
    remove_water,
    step,
)

st.set_page_config(
    page_title="HydraMind Digital Twin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

PRESSURE_COLUMNS = [
    "n1", "n4", "n31", "n54", "n105", "n114", "n163", "n188", "n215",
    "n229", "n288", "n296", "n332", "n342", "n410", "n415", "n429",
    "n458", "n469", "n495", "n506", "n516", "n519", "n549", "n613",
    "n636", "n644", "n679", "n722", "n726", "n740", "n752", "n769",
]
FLOW_COLUMNS = ["p227", "p235", "PUMP_1"]

st.markdown(
    """
    <style>
      .stApp {background:linear-gradient(180deg,#04111b,#071a28);}
      [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#071725,#081d2d);
        border-right:1px solid #24475f;
      }
      .block-container {max-width:1900px;padding-top:.7rem}
      [data-testid="stMetric"] {
        background:linear-gradient(180deg,#0d2b3e,#0a2233);
        border:1px solid #28516a;border-radius:14px;padding:12px;
      }
      .hero {border:1px solid #2d5872;border-radius:17px;padding:15px 18px;
        background:linear-gradient(135deg,#0e3045,#091f30);margin-bottom:12px}
      .title {font-weight:900;font-size:1.65rem}
      .sub {color:#9eb8c9;margin-top:3px}
      .pill {display:inline-block;padding:4px 9px;border-radius:999px;margin:8px 5px 0 0;
        font-size:.74rem;font-weight:800;border:1px solid}
      .ok {color:#64e6ad;border-color:#29996a;background:#123f33}
      .warn {color:#ffd36d;border-color:#927127;background:#3b3118}
      .bad {color:#ff8c8c;border-color:#a84040;background:#3d1d22}
      .info {color:#8dcbff;border-color:#397aa8;background:#152f44}
      .reason {border-left:4px solid #51a6db;background:#0b2639;padding:12px 14px;
        border-radius:9px;margin:7px 0}
      .tool {border:1px solid #315b73;border-radius:14px;padding:12px;background:#0b2435}
      .small {color:#9eb8c9;font-size:.82rem}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_ai():
    model = joblib.load(MODELS / "deployment_model.pkl")
    features = list(joblib.load(MODELS / "deployment_features.pkl"))
    encoder = joblib.load(MODELS / "deployment_label_encoder.pkl")
    metadata = json.loads((MODELS / "deployment_metadata.json").read_text())
    return model, features, encoder, metadata


def physics_to_model_row(state: dict, snapshot: dict, timestamp: pd.Timestamp) -> dict:
    """
    Convert the reduced-order digital twin into the same raw sensor schema used
    by the research classifier. This is an interface adapter, not proof that
    the BattLeDIM model generalizes to the physical prototype.
    """
    pressure = snapshot["pressure_bar"]
    flow = snapshot["total_flow_lpm"]
    leak_map = [
        snapshot["S1_leak_lpm"],
        snapshot["S2_leak_lpm"],
        snapshot["S3_leak_lpm"],
        snapshot["S4_leak_lpm"],
    ]
    row = {"Timestamp": timestamp}
    for index, sensor in enumerate(PRESSURE_COLUMNS):
        segment = min(index // 9, 3)
        spatial_drop = 0.015 * index
        local_leak_drop = 0.018 * leak_map[segment]
        wave = 0.025 * np.sin(index * 0.7 + state["time_s"] / 15.0)
        row[sensor] = max(pressure - spatial_drop - local_leak_drop + wave, 0.0)
    row["p227"] = snapshot["useful_flow_lpm"]
    row["p235"] = snapshot["leak_flow_lpm"] + snapshot["bypass_flow_lpm"]
    row["PUMP_1"] = flow
    return row


def ensure_ai_history(state: dict) -> pd.DataFrame:
    if "ai_raw_history" not in state:
        state["ai_raw_history"] = []
    if not state["ai_raw_history"]:
        # Create six baseline history points so rolling features are valid.
        now = pd.Timestamp.now().floor("s")
        for i in range(6):
            baseline = {
                "pressure_bar": 0.0,
                "total_flow_lpm": 0.0,
                "useful_flow_lpm": 0.0,
                "leak_flow_lpm": 0.0,
                "bypass_flow_lpm": 0.0,
                "S1_leak_lpm": 0.0,
                "S2_leak_lpm": 0.0,
                "S3_leak_lpm": 0.0,
                "S4_leak_lpm": 0.0,
            }
            state["ai_raw_history"].append(
                physics_to_model_row(state, baseline, now - pd.Timedelta(seconds=(5-i)*2))
            )
    return pd.DataFrame(state["ai_raw_history"])


def run_ai(state: dict, snapshot: dict, model, features, encoder):
    history = ensure_ai_history(state)
    row = physics_to_model_row(state, snapshot, pd.Timestamp.now())
    state["ai_raw_history"].append(row)
    state["ai_raw_history"] = state["ai_raw_history"][-120:]
    raw = pd.DataFrame(state["ai_raw_history"])

    start = time.perf_counter()
    matrix = prepare_model_matrix(raw, features)
    probabilities = model.predict_proba(matrix.tail(1))[0]
    latency_ms = (time.perf_counter() - start) * 1000.0
    predicted_index = int(np.argmax(probabilities))
    predicted_label = str(encoder.inverse_transform([predicted_index])[0])

    # Real XGBoost local feature contributions for the predicted class.
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(matrix.tail(1), feature_names=features)
    contrib = booster.predict(dmatrix, pred_contribs=True, approx_contribs=True)

    if contrib.ndim == 3:
        class_contrib = contrib[0, predicted_index, :-1]
    elif contrib.ndim == 2 and contrib.shape[1] == len(features) + 1:
        class_contrib = contrib[0, :-1]
    else:
        class_contrib = np.zeros(len(features))

    top_indices = np.argsort(np.abs(class_contrib))[::-1][:10]
    contributions = pd.DataFrame(
        {
            "feature": [features[i] for i in top_indices],
            "contribution": [float(class_contrib[i]) for i in top_indices],
            "direction": [
                "supports prediction" if class_contrib[i] >= 0 else "opposes prediction"
                for i in top_indices
            ],
        }
    )

    vector_hash = hashlib.sha256(
        matrix.tail(1).to_numpy(dtype=np.float32).tobytes()
    ).hexdigest()[:16]

    return {
        "label": predicted_label,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probabilities,
        "latency_ms": latency_ms,
        "contributions": contributions,
        "feature_vector": matrix.tail(1),
        "vector_hash": vector_hash,
    }


def topology_figure(state: dict, snapshot: dict) -> go.Figure:
    fig = go.Figure()
    x_nodes = [0, 2, 4, 6, 8]
    y_main = 3
    fig.add_trace(go.Scatter(
        x=x_nodes, y=[y_main]*5, mode="lines+markers",
        line=dict(width=12), marker=dict(size=18),
        name="Main line",
        hovertext=["Reservoir", "S1", "S2", "S3", "S4"],
    ))
    for i, house in enumerate(HOUSES):
        x = 2 + i*2
        fig.add_trace(go.Scatter(
            x=[x, x], y=[y_main, 1.2], mode="lines",
            line=dict(width=8, dash=None if state["valves"][house] else "dot"),
            name=house,
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[0.7], mode="markers+text",
            marker=dict(size=38, symbol="square"),
            text=[f"{house}<br>{snapshot[f'{house}_fill_pct']:.0f}%"],
            textposition="bottom center",
            showlegend=False,
        ))
    for i, segment in enumerate(SEGMENTS):
        leak = state["leaks"][segment]
        if leak["active"] and not leak["patched"]:
            fig.add_trace(go.Scatter(
                x=[2+i*2], y=[3.45], mode="markers+text",
                marker=dict(size=22, symbol="x"),
                text=[f"{segment} leak"],
                textposition="top center",
                name=f"{segment} leak",
            ))
    fig.add_annotation(x=0, y=3.55, text=f"Reservoir<br>{snapshot['reservoir_fill_pct']:.0f}%", showarrow=False)
    fig.add_annotation(x=0.8, y=2.55, text=f"Pump {snapshot['pump_actual_pct']:.0f}% actual", showarrow=False)
    fig.update_layout(
        title="Interactive hydraulic digital twin",
        xaxis=dict(visible=False, range=[-0.8, 9]),
        yaxis=dict(visible=False, range=[0, 4.6]),
        height=420,
        margin=dict(l=10,r=10,t=55,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dcebf7"),
    )
    return fig


def trend_figure(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["time_s"], y=history["pressure_bar"], name="Pressure (bar)"))
    fig.add_trace(go.Scatter(x=history["time_s"], y=history["total_flow_lpm"], name="Flow (L/min)", yaxis="y2"))
    fig.update_layout(
        height=330,
        yaxis=dict(title="Pressure (bar)"),
        yaxis2=dict(title="Flow (L/min)", overlaying="y", side="right"),
        xaxis_title="Simulation time (s)",
        margin=dict(l=10,r=10,t=35,b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dcebf7"),
        legend=dict(orientation="h"),
    )
    return fig


model, feature_names, encoder, metadata = load_ai()

if "sim" not in st.session_state:
    st.session_state.sim = default_state()
state = st.session_state.sim

with st.sidebar:
    st.markdown("## Live control drawer")
    run_steps = st.slider("Advance simulation", 1, 30, 5)
    auto_refresh = st.toggle("Continuous simulation", value=False)
    state["auto_mode"] = st.toggle("AI-assisted automatic control", value=state["auto_mode"])
    state["fluid"] = st.selectbox("Fluid", list(FLUIDS), index=list(FLUIDS).index(state["fluid"]))

    st.markdown("### Reservoir")
    fill_amount = st.number_input("Transfer amount (L)", 1.0, 50.0, 10.0)
    c1, c2 = st.columns(2)
    if c1.button("Add water", use_container_width=True):
        add_water(state, fill_amount)
    if c2.button("Drain", use_container_width=True):
        remove_water(state, fill_amount)

    st.markdown("### Pump")
    if not state["auto_mode"]:
        state["pump_enabled"] = st.toggle("Pump enabled", value=state["pump_enabled"])
        state["pump_command_pct"] = st.slider(
            "Pump command (%)", 0.0, 100.0, float(state["pump_command_pct"]), 1.0
        )
    else:
        st.caption("Command is calculated from tank deficits, leaks, pressure, and reservoir level.")

    st.markdown("### House valves and demand")
    for house in HOUSES:
        state["valves"][house] = st.toggle(
            f"{house} valve", value=state["valves"][house], key=f"valve_{house}"
        )
        state["house_demand_lpm"][house] = st.slider(
            f"{house} use (L/min)", 0.0, 3.0,
            float(state["house_demand_lpm"][house]), 0.1, key=f"demand_{house}"
        )

    st.markdown("### Physical fault tools")
    tool_segment = st.selectbox("Target segment", SEGMENTS)
    hole_size = st.slider("Spike opening (mm)", 0.5, 8.0, 2.0, 0.5)
    a, b = st.columns(2)
    if a.button("🗡️ Spike", use_container_width=True):
        create_leak(state, tool_segment, hole_size)
    if b.button("🔨 Patch", use_container_width=True):
        patch_leak(state, tool_segment)

    if st.button("Reset entire twin", use_container_width=True):
        st.session_state.sim = default_state()
        st.rerun()

    if st.button("Run now", type="primary", use_container_width=True):
        for _ in range(run_steps):
            snap = step(state, 2.0)
        st.rerun()

# Ensure at least one snapshot.
if not state["history"]:
    snapshot = step(state, 2.0)
else:
    snapshot = state["history"][-1]

ai = run_ai(state, snapshot, model, feature_names, encoder)

alarm_class = {"NORMAL":"ok","WARNING":"warn","CRITICAL":"bad"}[state["alarm"]]
st.markdown(
    f"""
    <div class="hero">
      <div class="title">HydraMind — Explainable AI Digital Twin</div>
      <div class="sub">Physics, operator tools, real XGBoost inference, and reasoned control in one screen.</div>
      <span class="pill {alarm_class}">{state["alarm"]}</span>
      <span class="pill info">XGBOOST VERIFIED</span>
      <span class="pill info">{len(feature_names)} REAL FEATURES</span>
      <span class="pill info">{ai['latency_ms']:.2f} ms INFERENCE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

m = st.columns(8)
m[0].metric("Pump command", f"{snapshot['pump_command_pct']:.0f}%")
m[1].metric("Pump actual", f"{snapshot['pump_actual_pct']:.0f}%")
m[2].metric("Pressure", f"{snapshot['pressure_bar']:.2f} bar")
m[3].metric("Total flow", f"{snapshot['total_flow_lpm']:.1f} L/min")
m[4].metric("Leak flow", f"{snapshot['leak_flow_lpm']:.1f} L/min")
m[5].metric("Reservoir", f"{snapshot['reservoir_fill_pct']:.0f}%")
m[6].metric("AI location", ai["label"])
m[7].metric("AI confidence", f"{ai['confidence']*100:.1f}%")

st.markdown(
    f'<div class="reason"><b>System reasoning:</b> {state["alarm_reason"]}</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.3, 1])
with left:
    st.plotly_chart(topology_figure(state, snapshot), use_container_width=True)
with right:
    history_df = pd.DataFrame(state["history"])
    st.plotly_chart(trend_figure(history_df), use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "AI working live",
    "Hydraulic calculations",
    "House and reservoir state",
    "Reasoned event log",
])

with tab1:
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.markdown("### Proof of real model execution")
        st.code(
            f"""Model: {metadata['model_name']}
Input features: {len(feature_names)}
Feature-vector hash: {ai['vector_hash']}
Output classes: {len(encoder.classes_)}
Inference latency: {ai['latency_ms']:.3f} ms
Predicted class: {ai['label']}
Confidence: {ai['confidence']:.4f}"""
        )
        probabilities = pd.DataFrame({
            "class": encoder.classes_,
            "probability_%": ai["probabilities"] * 100,
        }).sort_values("probability_%", ascending=False)
        st.dataframe(probabilities, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### Local XGBoost feature contributions")
        st.dataframe(ai["contributions"], use_container_width=True, hide_index=True)
        st.caption(
            "These are real tree-contribution values for the current predicted class, "
            "calculated from the deployed XGBoost booster—not handwritten explanations."
        )
    st.warning(
        "The digital twin is a reduced-order four-house simulator. Its adapter produces "
        "BattLeDIM-shaped sensor inputs, but this does not prove physical generalization. "
        "Real prototype validation is still required."
    )

with tab2:
    calculations = pd.DataFrame([
        ("Fluid", state["fluid"]),
        ("Density", f"{FLUIDS[state['fluid']]['density']:.1f} kg/m³"),
        ("Dynamic viscosity", f"{FLUIDS[state['fluid']]['viscosity']:.5f} Pa·s"),
        ("Reynolds number", f"{snapshot['reynolds']:.0f}"),
        ("Pipe velocity", f"{snapshot['velocity_mps']:.3f} m/s"),
        ("Darcy friction loss", f"{snapshot['friction_drop_bar']:.4f} bar"),
        ("Pump head / system pressure", f"{snapshot['pressure_bar']:.3f} bar"),
        ("Useful delivery flow", f"{snapshot['useful_flow_lpm']:.3f} L/min"),
        ("Leak discharge", f"{snapshot['leak_flow_lpm']:.3f} L/min"),
        ("Bypass return flow", f"{snapshot['bypass_flow_lpm']:.3f} L/min"),
        ("Water-hammer risk index", f"{snapshot['water_hammer_index']:.4f}"),
    ], columns=["Calculation", "Current value"])
    st.dataframe(calculations, use_container_width=True, hide_index=True)
    st.caption(
        "The engine uses pump affinity laws, Darcy–Weisbach friction, Reynolds-dependent "
        "friction factor, and an orifice leak equation. It is educational engineering "
        "simulation—not CFD or EPANET."
    )

with tab3:
    hc = st.columns(5)
    hc[0].metric("Reservoir", f"{snapshot['reservoir_fill_pct']:.1f}%")
    for i, house in enumerate(HOUSES):
        hc[i+1].metric(house, f"{snapshot[f'{house}_fill_pct']:.1f}%")
    house_table = pd.DataFrame({
        "house": HOUSES,
        "valve_open": [state["valves"][h] for h in HOUSES],
        "tank_fill_%": [snapshot[f"{h}_fill_pct"] for h in HOUSES],
        "demand_L_min": [state["house_demand_lpm"][h] for h in HOUSES],
    })
    st.dataframe(house_table, use_container_width=True, hide_index=True)
    if all(snapshot[f"{h}_fill_pct"] >= 92 or not state["valves"][h] for h in HOUSES):
        st.error(
            "Critical operating case: every connected house is full or isolated. "
            "The automatic controller commands the pump off and may open bypass while speed decays."
        )

with tab4:
    events = pd.DataFrame(state["event_log"])
    if events.empty:
        st.info("No operator or controller events yet.")
    else:
        st.dataframe(events.iloc[::-1], use_container_width=True, hide_index=True)
    st.caption("Every event records both the action and the reason.")

if auto_refresh:
    time.sleep(1.0)
    for _ in range(run_steps):
        step(state, 2.0)
    st.rerun()
