import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AquaGuard AI Digital Twin v2.2",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TIME_STEP_SECONDS = 1.0
MAX_HISTORY_ROWS = 1500

PRESSURE_IDS = [f"P{i}" for i in range(1, 9)]
FLOW_IDS = [f"FM{i}" for i in range(1, 5)]
BRANCH_VALVES = [f"V{i}" for i in range(1, 5)]
BYPASS_VALVES = ["BV1", "BV2"]

DEFAULT_PRESSURES = np.array([4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 3.6, 3.4], dtype=float)
DEFAULT_FLOWS = np.zeros(4, dtype=float)
DEFAULT_TANK_LEVELS = np.array([55.0, 62.0, 48.0, 70.0], dtype=float)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(17, 76, 122, 0.20), transparent 30%),
                linear-gradient(180deg, #07111d 0%, #0b1726 100%);
            color: #eaf4ff;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1c2d 0%, #07131f 100%);
            border-right: 1px solid #20364b;
        }

        [data-testid="stHeader"] {
            background: rgba(7, 17, 29, 0.75);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1800px;
        }

        h1, h2, h3 {
            color: #f4f9ff;
        }

        .hero {
            border: 1px solid #23415e;
            background: linear-gradient(135deg, rgba(12, 35, 55, 0.94), rgba(7, 22, 37, 0.94));
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }

        .hero-title {
            font-size: 1.62rem;
            font-weight: 800;
            letter-spacing: 0.3px;
        }

        .hero-subtitle {
            color: #9bb6ce;
            margin-top: 3px;
        }

        .status-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.7px;
            margin-right: 8px;
        }

        .chip-green { background: rgba(37, 196, 120, 0.17); color: #61e4a8; border: 1px solid #2ca46d; }
        .chip-yellow { background: rgba(246, 190, 55, 0.16); color: #ffd36b; border: 1px solid #b88a20; }
        .chip-red { background: rgba(240, 72, 72, 0.17); color: #ff8b8b; border: 1px solid #a33d3d; }
        .chip-blue { background: rgba(53, 139, 255, 0.16); color: #79b7ff; border: 1px solid #356fb4; }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(14, 38, 59, 0.94), rgba(9, 27, 44, 0.94));
            border: 1px solid #25435f;
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.16);
        }

        [data-testid="stMetricLabel"] {
            color: #9bb6ce;
        }

        [data-testid="stMetricValue"] {
            color: #f4f9ff;
        }

        .panel {
            background: linear-gradient(180deg, rgba(12, 31, 49, 0.96), rgba(8, 23, 38, 0.96));
            border: 1px solid #25435f;
            border-radius: 16px;
            padding: 15px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
        }

        .tiny {
            color: #8da9c1;
            font-size: 0.82rem;
        }

        .alarm {
            border-radius: 12px;
            padding: 10px 13px;
            margin: 7px 0;
            border-left: 4px solid;
            background: rgba(255,255,255,0.035);
        }

        .alarm-red { border-color: #ff5c5c; }
        .alarm-yellow { border-color: #ffc857; }
        .alarm-green { border-color: #42d392; }

        div[data-testid="stButton"] > button {
            border-radius: 10px;
            border: 1px solid #315a7a;
            background: #102b43;
            color: white;
            font-weight: 700;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #55a7e8;
            color: white;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #0c2236;
            border: 1px solid #23415e;
            border-radius: 9px;
            padding-left: 16px;
            padding-right: 16px;
        }

        .stTabs [aria-selected="true"] {
            background: #14456c;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STATE
# ============================================================

def initialize_state() -> None:
    defaults = {
        "running": False,
        "time_s": 0.0,
        "pressures": DEFAULT_PRESSURES.copy(),
        "flows": DEFAULT_FLOWS.copy(),
        "tank_levels": DEFAULT_TANK_LEVELS.copy(),
        "history": [],
        "event_log": [],
        "branch_valves": {v: True for v in BRANCH_VALVES},
        "bypass_valves": {v: False for v in BYPASS_VALVES},
        "ai_enabled": False,
        "emergency_shutdown": False,
        "demand_controlled_pump": True,
        "last_status": "NORMAL",
        "last_ai_action": "Monitoring only",
        "ai_steps": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_simulation() -> None:
    keys = [
        "running", "time_s", "pressures", "flows", "tank_levels", "history",
        "event_log", "branch_valves", "bypass_valves", "ai_enabled",
        "emergency_shutdown", "demand_controlled_pump", "last_status", "last_ai_action", "ai_steps"
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    initialize_state()


initialize_state()


# ============================================================
# DEPLOYMENT ARTIFACT CHECK
# ============================================================

@st.cache_resource
def inspect_deployment_artifacts() -> dict:
    files = {
        "model": MODELS_DIR / "deployment_model.pkl",
        "features": MODELS_DIR / "deployment_features.pkl",
        "encoder": MODELS_DIR / "deployment_label_encoder.pkl",
        "metadata": MODELS_DIR / "deployment_metadata.json",
    }
    status = {name: path.exists() for name, path in files.items()}
    metadata = {}
    if files["metadata"].exists():
        try:
            metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    return {"files": files, "status": status, "metadata": metadata}


artifacts = inspect_deployment_artifacts()
deployment_ready = all(artifacts["status"].values())


# ============================================================
# SIMULATION
# ============================================================

def system_status(pressures: np.ndarray, warning: float, danger: float) -> str:
    max_pressure = float(np.max(pressures))
    min_pressure = float(np.min(pressures))
    if max_pressure >= danger or min_pressure <= 0.8:
        return "CRITICAL"
    if max_pressure >= warning or min_pressure <= 1.8:
        return "WARNING"
    return "NORMAL"


def simulate_step(
    pump_speed: float,
    demand: np.ndarray,
    fault_type: str,
    fault_location: int,
    fault_severity: float,
    resistance: float,
    noise: float,
    warning: float,
    danger: float,
) -> None:
    if st.session_state.emergency_shutdown:
        pump_speed = 0.0

    branch_open = np.array(
        [1.0 if st.session_state.branch_valves[v] else 0.0 for v in BRANCH_VALVES]
    )

    total_requested_demand = float(np.sum(demand * branch_open))

    # In demand-controlled mode, the pump stops when no branch requests water.
    if st.session_state.demand_controlled_pump and total_requested_demand <= 0.0:
        pump_speed = 0.0

    # House delivery flow is proportional to BOTH demand and pump speed.
    # Therefore, 0% house demand produces 0 L/min under normal operation.
    hydraulic_efficiency = max(0.20, 1.0 - resistance / 125.0)
    max_branch_flow_l_min = 22.0
    flows_target = (
        max_branch_flow_l_min
        * (pump_speed / 100.0)
        * (demand / 100.0)
        * branch_open
        * hydraulic_efficiency
    )

    severity = fault_severity / 100.0
    idx = max(0, min(fault_location - 1, 3))

    pressure_target = np.linspace(
        1.3 + 0.055 * pump_speed,
        0.6 + 0.043 * pump_speed,
        8,
    )
    pressure_target -= resistance * np.linspace(0.003, 0.012, 8)

    if fault_type == "Leak":
        pressure_start = idx * 2
        pressure_target[pressure_start:] -= 4.0 * severity
        # A leak can create hydraulic loss even with no house demand.
        # It is tracked as a fault effect, not normal house consumption.
        flows_target[idx] += 3.5 * severity * (pump_speed / 100.0)
    elif fault_type == "Blockage":
        pressure_start = idx * 2
        pressure_target[: pressure_start + 1] += 3.8 * severity
        pressure_target[pressure_start + 1 :] -= 2.4 * severity
        flows_target[idx:] *= max(0.1, 1.0 - 0.75 * severity)
    elif fault_type == "Sensor Drift":
        pressure_start = idx * 2
        pressure_target[pressure_start] += 2.6 * severity
    elif fault_type == "Demand Surge":
        flows_target[idx] += 8.0 * severity
        pressure_target[idx * 2 :] -= 2.8 * severity

    if st.session_state.bypass_valves["BV1"]:
        pressure_target[:4] -= np.array([1.7, 1.45, 1.0, 0.7])
    if st.session_state.bypass_valves["BV2"]:
        pressure_target[4:] -= np.array([0.7, 1.0, 1.45, 1.7])

    # AI-assisted rule layer for the digital twin.
    # It does NOT claim to be the deployed XGBoost inference bridge yet.
    ai_steps = [
        "Sensor packet received",
        "Hydraulic features calculated",
        "Pressure-flow consistency checked",
    ]

    provisional_status = system_status(st.session_state.pressures, warning, danger)

    if st.session_state.ai_enabled:
        ai_steps.append("Deployment decision layer evaluated")
        if provisional_status == "CRITICAL":
            st.session_state.bypass_valves["BV1"] = True
            st.session_state.bypass_valves["BV2"] = True
            st.session_state.last_ai_action = "Opened both bypasses and issued critical alarm"
        elif provisional_status == "WARNING":
            if float(np.mean(st.session_state.pressures[:4])) >= float(np.mean(st.session_state.pressures[4:])):
                st.session_state.bypass_valves["BV1"] = True
                st.session_state.last_ai_action = "Opened upstream bypass BV1"
            else:
                st.session_state.bypass_valves["BV2"] = True
                st.session_state.last_ai_action = "Opened downstream bypass BV2"
        else:
            st.session_state.last_ai_action = "No corrective action required"
        ai_steps.append(st.session_state.last_ai_action)
    else:
        st.session_state.last_ai_action = "AI control disabled"

    st.session_state.ai_steps = ai_steps

    st.session_state.pressures += 0.20 * (pressure_target - st.session_state.pressures)
    st.session_state.pressures += np.random.normal(0.0, noise, size=8)
    st.session_state.pressures = np.clip(st.session_state.pressures, 0.0, 12.0)

    st.session_state.flows += 0.22 * (flows_target - st.session_state.flows)
    st.session_state.flows += np.random.normal(0.0, noise * 2.2, size=4)
    st.session_state.flows = np.clip(st.session_state.flows, 0.0, 35.0)

    inflow_factor = st.session_state.flows * 0.018
    usage_factor = demand * 0.004
    st.session_state.tank_levels += inflow_factor - usage_factor
    st.session_state.tank_levels = np.clip(st.session_state.tank_levels, 0.0, 100.0)

    st.session_state.time_s += TIME_STEP_SECONDS
    status = system_status(st.session_state.pressures, warning, danger)

    if status != st.session_state.last_status:
        st.session_state.event_log.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": status,
                "message": f"System status changed from {st.session_state.last_status} to {status}",
            },
        )
        st.session_state.last_status = status

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "time_s": round(st.session_state.time_s, 2),
        "status": status,
        "pump_speed_pct": pump_speed,
        "fault_type": fault_type,
        "fault_location": fault_location if fault_type != "None" else 0,
        "fault_severity_pct": fault_severity if fault_type != "None" else 0,
        "BV1_open": int(st.session_state.bypass_valves["BV1"]),
        "BV2_open": int(st.session_state.bypass_valves["BV2"]),
        "ai_enabled": int(st.session_state.ai_enabled),
        "ai_action": st.session_state.last_ai_action,
    }
    row.update({f"P{i+1}_bar": round(float(v), 4) for i, v in enumerate(st.session_state.pressures)})
    row.update({f"FM{i+1}_L_min": round(float(v), 4) for i, v in enumerate(st.session_state.flows)})
    row.update({f"Tank{i+1}_pct": round(float(v), 3) for i, v in enumerate(st.session_state.tank_levels)})

    st.session_state.history.append(row)
    st.session_state.history = st.session_state.history[-MAX_HISTORY_ROWS:]


# ============================================================
# SVG DIGITAL TWIN
# ============================================================

def valve_color(open_state: bool) -> str:
    return "#2ed58a" if open_state else "#f05252"


def pressure_color(value: float, warning: float, danger: float) -> str:
    if value >= danger or value <= 0.8:
        return "#ff5555"
    if value >= warning or value <= 1.8:
        return "#ffc857"
    return "#46b8ff"


def render_digital_twin(warning: float, danger: float) -> None:
    """Render a stable landscape digital twin using Plotly.

    Plotly avoids the raw-SVG parsing failure and the iframe black flashing
    seen in the previous versions.
    """

    p = st.session_state.pressures
    f = st.session_state.flows
    t = st.session_state.tank_levels
    bv1 = st.session_state.bypass_valves["BV1"]
    bv2 = st.session_state.bypass_valves["BV2"]
    valves = st.session_state.branch_valves

    active_demand = any(
        valves[f"V{i+1}"]
        and st.session_state.get(f"_demand_{i+1}", 0.0) > 0
        for i in range(4)
    )
    pump_running = (
        st.session_state.running
        and not st.session_state.emergency_shutdown
        and (active_demand or not st.session_state.demand_controlled_pump)
    )

    fig = go.Figure()

    # Canvas
    fig.update_xaxes(range=[0, 112], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0, 68], visible=False, fixedrange=True, scaleanchor="x")
    fig.update_layout(
        height=690,
        margin=dict(l=8, r=8, t=55, b=8),
        paper_bgcolor="#081520",
        plot_bgcolor="#081520",
        showlegend=False,
        dragmode=False,
        title=dict(
            text=(
                "<b>LIVE WATER DISTRIBUTION DIGITAL TWIN</b>"
                "<br><span style='font-size:12px;color:#83a3bc'>"
                "Main ring • branch delivery • bypass relief • sensing • edge control • AI decision path"
                "</span>"
            ),
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(color="#eaf5ff", size=17),
        ),
    )

    pipe = "#b8c5ce"
    pipe_edge = "#6c8294"
    water = "#28aef4"
    text = "#eaf5ff"
    muted = "#8eb0c7"

    def line(x0, y0, x1, y1, width=16, color=pipe, dash=None, layer="below"):
        fig.add_shape(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=color, width=width, dash=dash),
            layer=layer,
        )

    def box(x0, y0, x1, y1, fill, border="#315a78", radius=0):
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=fill,
            line=dict(color=border, width=2),
            layer="below",
        )

    def label(x, y, value, size=12, color=text, bold=False, angle=0):
        rendered = f"<b>{value}</b>" if bold else value
        fig.add_annotation(
            x=x, y=y, text=rendered, showarrow=False,
            font=dict(size=size, color=color),
            textangle=angle,
            xanchor="center", yanchor="middle",
        )

    # Reservoir
    box(2, 35, 13, 52, "#0f6ca3", "#47baf2")
    fig.add_shape(type="circle", x0=2, y0=49.5, x1=13, y1=54.2,
                  fillcolor="#168ed0", line=dict(color="#47baf2", width=2))
    fig.add_shape(type="circle", x0=2, y0=32.8, x1=13, y1=37.5,
                  fillcolor="#0b5b8d", line=dict(color="#0b5b8d", width=1))
    label(7.5, 31.6, "RESERVOIR", 13, text, True)

    # Pump
    line(13, 43, 18, 43, 14)
    fig.add_shape(type="circle", x0=18, y0=39, x1=26, y1=47,
                  fillcolor="#da8d18", line=dict(color="#ffd27b", width=3))
    fig.add_shape(type="circle", x0=20.6, y0=41.6, x1=23.4, y1=44.4,
                  fillcolor="#5b3810", line=dict(color="#5b3810"))
    label(22, 36.7, "PUMP", 13, text, True)
    label(22, 34.5, "RUNNING" if pump_running else "STOPPED", 11,
          "#55d89a" if pump_running else "#f07878", True)

    # Check valve
    line(26, 43, 29, 43, 14)
    fig.add_shape(
        type="path",
        path="M 29 43 L 31 45 L 31 41 Z",
        fillcolor="#55d89a",
        line=dict(color="#55d89a"),
    )
    line(31.3, 40.7, 31.3, 45.3, 3, "#d4edf9")
    label(30.2, 47.5, "CHECK VALVE", 10, muted, True)

    # Main ring
    line(32, 55, 88, 55, 16)
    line(32, 32, 88, 32, 16)
    line(88, 32, 88, 55, 16)
    line(32, 43, 32, 55, 16)

    # Pipe highlights
    line(32, 55.5, 88, 55.5, 3, "#eef7fc")
    line(32, 32.5, 88, 32.5, 3, "#eef7fc")
    line(88.5, 32, 88.5, 55, 3, "#eef7fc")

    # Flow tracks
    line(32, 55, 88, 55, 3, water, "dash")
    line(88, 55, 88, 32, 3, water, "dash")
    line(88, 32, 32, 32, 3, water, "dash")

    # Bypasses
    bypasses = [(45, "BV1", bv1), (70, "BV2", bv2)]
    for x, name, opened in bypasses:
        line(x, 32, x, 55, 14)
        fig.add_shape(
            type="circle", x0=x-1.8, y0=41.7, x1=x+1.8, y1=45.3,
            fillcolor=valve_color(opened),
            line=dict(color="#e9f5fb", width=2),
        )
        label(x, 43.5, name, 11, "#07111d", True)
        label(x, 39.8, "OPEN" if opened else "CLOSED", 10,
              "#55d89a" if opened else "#f07878", True)

    # Pressure sensors
    sensor_positions = [
        (36, 55, 60.5), (50, 55, 60.5), (65, 55, 60.5), (83, 55, 60.5),
        (36, 32, 36.8), (50, 32, 36.8), (65, 32, 36.8), (83, 32, 36.8),
    ]
    for i, (x, pipe_y, sensor_y) in enumerate(sensor_positions):
        line(x, pipe_y, x, sensor_y - 1.5, 3, "#8fa8bc")
        color = pressure_color(float(p[i]), warning, danger)
        box(x-2.2, sensor_y-1.4, x+2.2, sensor_y+1.5, color, color)
        label(x, sensor_y, f"P{i+1}", 11, "#07111d", True)
        label(x, sensor_y + 3.1, f"{p[i]:.2f} bar", 10, text, True)

    # Branches, valves, flowmeters, houses and tanks
    house_x = [36, 50, 65, 83]
    for i, x in enumerate(house_x):
        opened = valves[f"V{i+1}"]
        line(x, 32, x, 15.5, 12)

        fig.add_shape(
            type="circle", x0=x-1.5, y0=23.2, x1=x+1.5, y1=26.2,
            fillcolor=valve_color(opened),
            line=dict(color="#e9f5fb", width=2),
        )
        label(x, 24.7, f"V{i+1}", 10, "#07111d", True)

        box(x-2.6, 18.7, x+2.6, 21.7, "#0879a8", "#58c6f2")
        label(x, 20.2, f"FM{i+1}", 10, "#06111a", True)
        label(x, 17.0, f"{f[i]:.1f} L/min", 10, "#a8d9f5", True)

        # House body and roof
        fig.add_shape(
            type="path",
            path=(
                f"M {x-4.1} 10.7 L {x} 14.0 L {x+4.1} 10.7 "
                f"L {x+4.1} 3.1 L {x-4.1} 3.1 Z"
            ),
            fillcolor="#112a40",
            line=dict(color="#8bb4d1", width=2),
        )

        # Mini tank
        tank_x0, tank_x1 = x-2.0, x+2.0
        tank_y0, tank_y1 = 5.0, 10.7
        level_y = tank_y0 + (tank_y1 - tank_y0) * float(t[i]) / 100.0
        box(tank_x0, tank_y0, tank_x1, tank_y1, "rgba(0,0,0,0)", "#b5d4e8")
        box(tank_x0, tank_y0, tank_x1, level_y, "#1b9ee8", "#1b9ee8")
        label(x, 1.8, f"HOUSE {i+1}", 11, text, True)
        label(x, 0.2, f"Tank {t[i]:.0f}%", 9, muted, True)

    # Edge and AI stack
    stack_x0, stack_x1 = 94, 110
    stack = [
        (51, 59, "#102f4a", "RASPBERRY PI", ["Data logging", "Feature pipeline"]),
        (39, 48, "#173a58", "XGBOOST AI", [
            "ENABLED" if st.session_state.ai_enabled else "DISABLED",
            "Leak localization",
        ]),
        (27, 35, "#102f4a", "ESP32 CONTROL", ["Valve commands", "Manual override"]),
    ]
    for y0, y1, fill, title, lines in stack:
        box(stack_x0, y0, stack_x1, y1, fill, "#3f7ba5")
        label(102, y1-2.0, title, 11, text, True)
        label(102, y1-4.2, lines[0], 9, muted)
        label(102, y1-6.0, lines[1], 9, muted)

    # Technology links
    line(102, 51, 102, 48, 2, water, "dot", "above")
    line(102, 39, 102, 35, 2, water, "dot", "above")
    fig.add_annotation(
        x=88, y=43, ax=94, ay=31,
        text="", showarrow=True,
        arrowhead=3, arrowwidth=2, arrowcolor=water,
    )

    # Operating status and timing
    label(102, 22, f"Simulation: {st.session_state.time_s:.1f} s", 10, muted, True)
    label(102, 19.5, f"AI action: {st.session_state.last_ai_action[:25]}", 9, muted)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "staticPlot": True,
            "responsive": True,
        },
        key="stable_digital_twin",
    )


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.markdown("## Control Center")

st.sidebar.caption("The single AI switch is located in the main control bar.")

pump_speed = st.sidebar.slider("Maximum pump speed (%)", 0, 100, 68, 1)
st.session_state.demand_controlled_pump = st.sidebar.toggle(
    "Demand-controlled pump",
    value=st.session_state.demand_controlled_pump,
    help="Automatically stops the pump when all active house demands are 0%.",
)
resistance = st.sidebar.slider("Pipe resistance (%)", 0, 100, 28, 1)
noise = st.sidebar.slider("Sensor noise", 0.0, 0.12, 0.018, 0.002, format="%.3f")

warning_limit = st.sidebar.slider("Warning pressure (bar)", 3.0, 9.0, 6.5, 0.1)
danger_limit = st.sidebar.slider(
    "Critical pressure (bar)",
    warning_limit + 0.1,
    11.0,
    max(8.0, warning_limit + 0.1),
    0.1,
)

st.sidebar.divider()
st.sidebar.markdown("### House demand")
demands = np.array(
    [
        st.sidebar.slider("House 1 demand (%)", 0, 100, 48),
        st.sidebar.slider("House 2 demand (%)", 0, 100, 42),
        st.sidebar.slider("House 3 demand (%)", 0, 100, 37),
        st.sidebar.slider("House 4 demand (%)", 0, 100, 45),
    ],
    dtype=float,
)

st.sidebar.divider()
st.sidebar.markdown("### Fault injection")
fault_type = st.sidebar.selectbox(
    "Scenario",
    ["None", "Leak", "Blockage", "Sensor Drift", "Demand Surge"],
)
fault_location = st.sidebar.selectbox(
    "Location",
    [1, 2, 3, 4],
    format_func=lambda x: f"Branch / zone {x}",
    disabled=fault_type == "None",
)
fault_severity = st.sidebar.slider(
    "Severity (%)",
    0,
    100,
    55 if fault_type != "None" else 0,
    5,
    disabled=fault_type == "None",
)

st.sidebar.divider()
st.sidebar.markdown("### Safety")
st.session_state.emergency_shutdown = st.sidebar.toggle(
    "Emergency shutdown",
    value=st.session_state.emergency_shutdown,
)

st.sidebar.info(
    "Normal behavior: house flow becomes 0 L/min at 0% demand. "
    "With demand-controlled pump enabled, the pump also stops when total demand is zero."
)


# Store demand values for the SVG pump-state indicator.
for _i, _value in enumerate(demands, start=1):
    st.session_state[f"_demand_{_i}"] = float(_value)


# ============================================================
# HEADER
# ============================================================

current_status = system_status(st.session_state.pressures, warning_limit, danger_limit)
chip_class = {
    "NORMAL": "chip-green",
    "WARNING": "chip-yellow",
    "CRITICAL": "chip-red",
}[current_status]

st.markdown(
    f"""
    <div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:20px;">
            <div>
                <div class="hero-title">AquaGuard AI — Smart Water Distribution Digital Twin v2.2</div>
                <div class="hero-subtitle">Real-time hydraulic simulation, fault injection, AI-assisted control and system explainability</div>
            </div>
            <div style="text-align:right;">
                <span class="status-chip {chip_class}">{current_status}</span>
                <span class="status-chip chip-blue">{'DEPLOYMENT FILES READY' if deployment_ready else 'DEPLOYMENT FILES INCOMPLETE'}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PRIMARY CONTROLS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("▶ Start", use_container_width=True, type="primary"):
        st.session_state.running = True
with c2:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.running = False
with c3:
    step_clicked = st.button("⏭ Step", use_container_width=True)
with c4:
    if st.button("↺ Reset", use_container_width=True):
        reset_simulation()
        st.rerun()

ai_left, ai_right = st.columns([0.24, 0.76])
with ai_left:
    st.toggle(
        "Enable AI-powered control",
        key="ai_enabled",
        help="Single master control for AI-assisted operation.",
    )
with ai_right:
    st.caption(
        "When enabled, the decision layer can open bypass valves in response "
        "to warning and critical hydraulic conditions."
    )


# ============================================================
# LIVE UPDATE
# ============================================================

@st.fragment(run_every=TIME_STEP_SECONDS)
def live_view() -> None:
    if st.session_state.running:
        simulate_step(
            pump_speed=pump_speed,
            demand=demands,
            fault_type=fault_type,
            fault_location=fault_location,
            fault_severity=fault_severity,
            resistance=resistance,
            noise=noise,
            warning=warning_limit,
            danger=danger_limit,
        )

    p = st.session_state.pressures
    f = st.session_state.flows
    t = st.session_state.tank_levels
    status = system_status(p, warning_limit, danger_limit)

    metrics = st.columns(7)
    metrics[0].metric("System", status)
    metrics[1].metric("Max pressure", f"{np.max(p):.2f} bar")
    metrics[2].metric("Min pressure", f"{np.min(p):.2f} bar")
    metrics[3].metric("Total flow", f"{np.sum(f):.1f} L/min")
    effective_pump_display = (
        "OFF"
        if st.session_state.emergency_shutdown
        or (
            st.session_state.demand_controlled_pump
            and float(np.sum(demands)) <= 0.0
        )
        else f"{pump_speed}%"
    )
    metrics[4].metric("Pump", effective_pump_display)
    metrics[5].metric("BV1", "OPEN" if st.session_state.bypass_valves["BV1"] else "CLOSED")
    metrics[6].metric("BV2", "OPEN" if st.session_state.bypass_valves["BV2"] else "CLOSED")

    tab_overview, tab_controls, tab_ai, tab_analytics, tab_logs, tab_settings = st.tabs(
        ["Digital Twin", "Manual Controls", "AI Engine", "Analytics", "Event Log", "Settings"]
    )

    with tab_overview:
        render_digital_twin(warning_limit, danger_limit)

        lower_left, lower_right = st.columns([1.15, 0.85])
        with lower_left:
            st.markdown("### Live sensor summary")
            sensor_df = pd.DataFrame(
                {
                    "Sensor": PRESSURE_IDS,
                    "Pressure (bar)": np.round(p, 3),
                    "State": [
                        "CRITICAL" if x >= danger_limit or x <= 0.8
                        else "WARNING" if x >= warning_limit or x <= 1.8
                        else "NORMAL"
                        for x in p
                    ],
                }
            )
            st.dataframe(sensor_df, use_container_width=True, hide_index=True)

        with lower_right:
            st.markdown("### Active alarms")
            if status == "CRITICAL":
                st.markdown(
                    '<div class="alarm alarm-red"><b>Critical hydraulic condition</b><br><span class="tiny">Immediate operator review required.</span></div>',
                    unsafe_allow_html=True,
                )
            elif status == "WARNING":
                st.markdown(
                    '<div class="alarm alarm-yellow"><b>Hydraulic warning</b><br><span class="tiny">Pressure or flow is outside the preferred operating envelope.</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alarm alarm-green"><b>No active alarms</b><br><span class="tiny">All monitored values are inside the simulated operating envelope.</span></div>',
                    unsafe_allow_html=True,
                )
            if fault_type != "None":
                st.markdown(
                    f'<div class="alarm alarm-yellow"><b>Injected scenario: {fault_type}</b><br><span class="tiny">Zone {fault_location}, severity {fault_severity}%.</span></div>',
                    unsafe_allow_html=True,
                )

    with tab_controls:
        st.markdown("### Manual component controls")
        st.caption("All bypasses begin closed. Branch valves begin open.")

        cols = st.columns(6)
        for i, valve in enumerate(BRANCH_VALVES):
            with cols[i]:
                new_state = st.toggle(
                    f"{valve} — House {i+1}",
                    value=st.session_state.branch_valves[valve],
                    key=f"control_{valve}",
                )
                st.session_state.branch_valves[valve] = new_state

        with cols[4]:
            st.session_state.bypass_valves["BV1"] = st.toggle(
                "BV1 upstream bypass",
                value=st.session_state.bypass_valves["BV1"],
                key="control_BV1",
            )
        with cols[5]:
            st.session_state.bypass_valves["BV2"] = st.toggle(
                "BV2 downstream bypass",
                value=st.session_state.bypass_valves["BV2"],
                key="control_BV2",
            )

        st.markdown("### Component readings")
        component_df = pd.DataFrame(
            {
                "Component": PRESSURE_IDS + FLOW_IDS + [f"Tank {i}" for i in range(1, 5)],
                "Type": ["Pressure transducer"] * 8 + ["Flow meter"] * 4 + ["Level sensor"] * 4,
                "Value": (
                    [f"{x:.2f} bar" for x in p]
                    + [f"{x:.2f} L/min" for x in f]
                    + [f"{x:.1f}%" for x in t]
                ),
            }
        )
        st.dataframe(component_df, use_container_width=True, hide_index=True)

    with tab_ai:
        left, right = st.columns([0.75, 1.25])

        with left:
            st.markdown("### AI status")
            st.metric("Control state", "ENABLED" if st.session_state.ai_enabled else "DISABLED")
            st.metric("Latest action", st.session_state.last_ai_action)
            st.metric("Model package", "XGBoost deployment" if deployment_ready else "Missing artifact(s)")
            st.metric("Input feature target", "205 engineered features")

            if not deployment_ready:
                missing = [name for name, exists in artifacts["status"].items() if not exists]
                st.warning("Missing: " + ", ".join(missing))

        with right:
            st.markdown("### Decision trace")
            if st.session_state.ai_steps:
                for number, text in enumerate(st.session_state.ai_steps, start=1):
                    st.markdown(f"**{number}.** {text}")
            else:
                st.info("Start the simulation to populate the AI decision trace.")

            st.markdown("### Important integration status")
            st.info(
                "This interface currently runs the hydraulic digital twin and an AI-assisted control layer. "
                "The saved XGBoost model is detected, but exact model inference must be connected through the same "
                "205-feature engineering pipeline used during training. The app deliberately does not fabricate "
                "missing model features."
            )

    with tab_analytics:
        if st.session_state.history:
            data = pd.DataFrame(st.session_state.history)

            st.markdown("### Pressure trends")
            st.line_chart(
                data.tail(500),
                x="time_s",
                y=[f"P{i}_bar" for i in range(1, 9)],
                height=360,
            )

            st.markdown("### House flow trends")
            st.line_chart(
                data.tail(500),
                x="time_s",
                y=[f"FM{i}_L_min" for i in range(1, 5)],
                height=330,
            )

            st.markdown("### Tank levels")
            st.line_chart(
                data.tail(500),
                x="time_s",
                y=[f"Tank{i}_pct" for i in range(1, 5)],
                height=330,
            )

            a, b, c = st.columns(3)
            a.metric("Recorded samples", len(data))
            b.metric("Average total flow", f"{data[[f'FM{i}_L_min' for i in range(1,5)]].sum(axis=1).mean():.2f} L/min")
            c.metric("Critical samples", int((data["status"] == "CRITICAL").sum()))

            csv_bytes = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download simulation dataset",
                csv_bytes,
                "aquaguard_simulation.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.info("Start or step the simulation to generate analytics.")

    with tab_logs:
        st.markdown("### Event timeline")
        if st.session_state.event_log:
            st.dataframe(
                pd.DataFrame(st.session_state.event_log),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No state-change events have been recorded.")

    with tab_settings:
        st.markdown("### Interface and engineering settings")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.selectbox("Pressure unit", ["bar", "kPa", "psi"], disabled=True)
            st.caption("Unit conversion will be enabled after the base dashboard is validated.")
        with s2:
            st.selectbox("Operator role", ["Engineer", "Operator", "Viewer"])
            st.caption("A secure database-backed account system should be added only after the core application works.")
        with s3:
            st.selectbox("Theme", ["Industrial dark"], disabled=True)
            st.caption("This first version uses a fixed control-room theme.")


live_view()


if step_clicked:
    st.session_state.running = False
    simulate_step(
        pump_speed=pump_speed,
        demand=demands,
        fault_type=fault_type,
        fault_location=fault_location,
        fault_severity=fault_severity,
        resistance=resistance,
        noise=noise,
        warning=warning_limit,
        danger=danger_limit,
    )
    st.rerun()


st.caption(
    "Scientific note: the hydraulic behavior is a software digital-twin approximation. "
    "Physical prototype measurements are still required for real-world validation."
)