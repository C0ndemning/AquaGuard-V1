from __future__ import annotations

from dataclasses import dataclass, asdict
from math import pi, sqrt
from typing import Any

import numpy as np


FLUIDS = {
    "Water": {"density": 998.0, "viscosity": 0.00100},
    "Light oil": {"density": 850.0, "viscosity": 0.01800},
    "Glycol mix": {"density": 1040.0, "viscosity": 0.00450},
}

SEGMENTS = ["S1", "S2", "S3", "S4"]
HOUSES = ["House 1", "House 2", "House 3", "House 4"]


@dataclass
class Leak:
    active: bool = False
    severity_mm: float = 0.0
    patched: bool = False


def default_state() -> dict[str, Any]:
    return {
        "time_s": 0.0,
        "fluid": "Water",
        "reservoir_capacity_l": 220.0,
        "reservoir_volume_l": 165.0,
        "pump_command_pct": 0.0,
        "pump_actual_pct": 0.0,
        "pump_enabled": False,
        "auto_mode": True,
        "bypass_open": False,
        "valves": {house: True for house in HOUSES},
        "house_capacity_l": {house: 30.0 for house in HOUSES},
        "house_volume_l": {
            "House 1": 12.0,
            "House 2": 17.0,
            "House 3": 23.0,
            "House 4": 8.0,
        },
        "house_demand_lpm": {
            "House 1": 1.2,
            "House 2": 0.8,
            "House 3": 1.5,
            "House 4": 1.0,
        },
        "leaks": {segment: asdict(Leak()) for segment in SEGMENTS},
        "last_total_flow_lpm": 0.0,
        "last_pressure_bar": 0.0,
        "event_log": [],
        "history": [],
        "alarm": "NORMAL",
        "alarm_reason": "System initialized.",
    }


def log_event(state: dict[str, Any], action: str, reason: str, severity: str = "INFO") -> None:
    state["event_log"].append(
        {
            "time_s": round(float(state["time_s"]), 1),
            "severity": severity,
            "action": action,
            "reason": reason,
        }
    )
    state["event_log"] = state["event_log"][-200:]


def fluid_properties(state: dict[str, Any]) -> tuple[float, float]:
    props = FLUIDS[state["fluid"]]
    return float(props["density"]), float(props["viscosity"])


def reynolds_number(flow_lpm: float, diameter_m: float, density: float, viscosity: float) -> float:
    q = max(flow_lpm, 0.0) / 1000.0 / 60.0
    area = pi * diameter_m**2 / 4.0
    velocity = q / max(area, 1e-9)
    return density * velocity * diameter_m / max(viscosity, 1e-9)


def friction_factor(reynolds: float) -> float:
    if reynolds <= 0:
        return 0.0
    if reynolds < 2300:
        return 64.0 / reynolds
    return 0.3164 / reynolds**0.25


def darcy_pressure_drop_bar(
    flow_lpm: float,
    length_m: float,
    diameter_m: float,
    density: float,
    viscosity: float,
) -> tuple[float, float, float]:
    q = max(flow_lpm, 0.0) / 1000.0 / 60.0
    area = pi * diameter_m**2 / 4.0
    velocity = q / max(area, 1e-9)
    re = reynolds_number(flow_lpm, diameter_m, density, viscosity)
    f = friction_factor(re)
    dp_pa = f * (length_m / diameter_m) * density * velocity**2 / 2.0
    return dp_pa / 100000.0, re, velocity


def pump_head_bar(speed_pct: float, density: float) -> float:
    # Nominal 4.2 bar at 100% for water; pump affinity law H ∝ N².
    water_density = 998.0
    return 4.2 * (speed_pct / 100.0) ** 2 * (density / water_density)


def leak_flow_lpm(pressure_bar: float, hole_mm: float, density: float) -> float:
    if pressure_bar <= 0 or hole_mm <= 0:
        return 0.0
    cd = 0.62
    area = pi * (hole_mm / 1000.0) ** 2 / 4.0
    q_m3s = cd * area * sqrt(2.0 * pressure_bar * 100000.0 / density)
    return q_m3s * 1000.0 * 60.0


def house_fill_pct(state: dict[str, Any], house: str) -> float:
    return 100.0 * state["house_volume_l"][house] / state["house_capacity_l"][house]


def automatic_controller(state: dict[str, Any]) -> None:
    fills = {house: house_fill_pct(state, house) for house in HOUSES}
    needing_water = [
        house for house in HOUSES
        if state["valves"][house] and fills[house] < 92.0
    ]
    full_or_closed = all(
        (fills[house] >= 92.0) or (not state["valves"][house])
        for house in HOUSES
    )
    active_leaks = [
        segment for segment in SEGMENTS
        if state["leaks"][segment]["active"] and not state["leaks"][segment]["patched"]
    ]

    if full_or_closed and not active_leaks:
        if state["pump_enabled"] or state["pump_command_pct"] > 0:
            log_event(
                state,
                "Pump stop command",
                "All connected house tanks are sufficiently full or isolated; continued pumping would raise pressure without useful delivery.",
                "WARNING",
            )
        state["pump_enabled"] = False
        state["pump_command_pct"] = 0.0
        state["bypass_open"] = state["last_pressure_bar"] > 2.5
        return

    if state["reservoir_volume_l"] < 12.0:
        state["pump_enabled"] = False
        state["pump_command_pct"] = 0.0
        state["bypass_open"] = False
        log_event(
            state,
            "Low-level shutdown",
            "Reservoir volume is below the pump-protection limit, so the controller prevents dry running.",
            "CRITICAL",
        )
        return

    if needing_water:
        average_deficit = np.mean([100.0 - fills[h] for h in needing_water])
        command = float(np.clip(35.0 + 0.65 * average_deficit, 35.0, 88.0))
        if active_leaks:
            command = min(command, 48.0)
        state["pump_enabled"] = True
        state["pump_command_pct"] = command
        state["bypass_open"] = False
    elif active_leaks:
        state["pump_enabled"] = False
        state["pump_command_pct"] = 0.0
        state["bypass_open"] = True


def step(state: dict[str, Any], dt_s: float = 2.0) -> dict[str, Any]:
    if state["auto_mode"]:
        automatic_controller(state)

    target_speed = state["pump_command_pct"] if state["pump_enabled"] else 0.0

    # Pump inertia: actual speed approaches command, and decays after stop.
    response_tau_s = 3.5
    alpha = 1.0 - np.exp(-dt_s / response_tau_s)
    previous_speed = float(state["pump_actual_pct"])
    state["pump_actual_pct"] += (target_speed - state["pump_actual_pct"]) * alpha
    if state["pump_actual_pct"] < 0.25 and target_speed == 0:
        state["pump_actual_pct"] = 0.0

    density, viscosity = fluid_properties(state)

    fills = {house: house_fill_pct(state, house) for house in HOUSES}
    requested = {}
    for house in HOUSES:
        deficit_factor = np.clip((100.0 - fills[house]) / 35.0, 0.0, 1.0)
        requested[house] = (
            (3.0 + 6.0 * deficit_factor)
            if state["valves"][house] and fills[house] < 99.5
            else 0.0
        )

    useful_flow_lpm = sum(requested.values()) * (state["pump_actual_pct"] / 100.0)
    preliminary_head = pump_head_bar(state["pump_actual_pct"], density)

    leak_flows = {}
    for segment in SEGMENTS:
        leak = state["leaks"][segment]
        leak_flows[segment] = (
            leak_flow_lpm(preliminary_head, leak["severity_mm"], density)
            if leak["active"] and not leak["patched"]
            else 0.0
        )

    bypass_flow_lpm = 0.0
    if state["bypass_open"] and preliminary_head > 0.3:
        bypass_flow_lpm = 8.0 * sqrt(preliminary_head)

    total_flow_lpm = useful_flow_lpm + sum(leak_flows.values()) + bypass_flow_lpm

    pipe_length_m = 5.0
    diameter_m = 0.020
    pressure_drop_bar, re, velocity = darcy_pressure_drop_bar(
        total_flow_lpm, pipe_length_m, diameter_m, density, viscosity
    )

    # With closed valves and a running pump, static pressure rises toward pump head.
    pressure_bar = max(preliminary_head - pressure_drop_bar, 0.0)
    if useful_flow_lpm < 0.2 and state["pump_actual_pct"] > 5:
        pressure_bar = preliminary_head * 0.98

    # Reservoir loses delivered and leaked water. Bypass returns to reservoir.
    delivered_l = useful_flow_lpm * dt_s / 60.0
    leaked_l = sum(leak_flows.values()) * dt_s / 60.0

    state["reservoir_volume_l"] = float(np.clip(
        state["reservoir_volume_l"] - delivered_l - leaked_l,
        0.0,
        state["reservoir_capacity_l"],
    ))

    if useful_flow_lpm > 0:
        total_requested = max(sum(requested.values()), 1e-9)
        for house in HOUSES:
            branch_flow = useful_flow_lpm * requested[house] / total_requested
            inflow_l = branch_flow * dt_s / 60.0
            use_l = state["house_demand_lpm"][house] * dt_s / 60.0
            state["house_volume_l"][house] = float(np.clip(
                state["house_volume_l"][house] + inflow_l - use_l,
                0.0,
                state["house_capacity_l"][house],
            ))
    else:
        for house in HOUSES:
            use_l = state["house_demand_lpm"][house] * dt_s / 60.0
            state["house_volume_l"][house] = float(np.clip(
                state["house_volume_l"][house] - use_l,
                0.0,
                state["house_capacity_l"][house],
            ))

    flow_change = total_flow_lpm - float(state["last_total_flow_lpm"])
    water_hammer_index = abs(flow_change) * density * max(velocity, 0.01) / 100000.0

    state["last_total_flow_lpm"] = float(total_flow_lpm)
    state["last_pressure_bar"] = float(pressure_bar)
    state["time_s"] += dt_s

    active_leaks = [s for s, q in leak_flows.items() if q > 0]
    all_satisfied = all(
        house_fill_pct(state, h) >= 92.0 or not state["valves"][h]
        for h in HOUSES
    )

    if pressure_bar > 4.0:
        state["alarm"] = "CRITICAL"
        state["alarm_reason"] = "Pressure exceeds the 4.0 bar safety limit."
    elif all_satisfied and state["pump_actual_pct"] > 10.0:
        state["alarm"] = "CRITICAL"
        state["alarm_reason"] = "All house tanks are satisfied while the pump is still spinning; stop or bypass is required."
    elif active_leaks:
        state["alarm"] = "WARNING"
        state["alarm_reason"] = f"Hydraulic loss is active in {', '.join(active_leaks)}."
    elif water_hammer_index > 0.35:
        state["alarm"] = "WARNING"
        state["alarm_reason"] = "Rapid flow change indicates water-hammer risk."
    else:
        state["alarm"] = "NORMAL"
        state["alarm_reason"] = "Hydraulic values remain inside configured operating limits."

    snapshot = {
        "time_s": state["time_s"],
        "pump_command_pct": state["pump_command_pct"],
        "pump_actual_pct": state["pump_actual_pct"],
        "pressure_bar": pressure_bar,
        "total_flow_lpm": total_flow_lpm,
        "useful_flow_lpm": useful_flow_lpm,
        "leak_flow_lpm": sum(leak_flows.values()),
        "bypass_flow_lpm": bypass_flow_lpm,
        "reservoir_fill_pct": 100.0 * state["reservoir_volume_l"] / state["reservoir_capacity_l"],
        "reynolds": re,
        "velocity_mps": velocity,
        "friction_drop_bar": pressure_drop_bar,
        "water_hammer_index": water_hammer_index,
        **{f"{h}_fill_pct": house_fill_pct(state, h) for h in HOUSES},
        **{f"{s}_leak_lpm": leak_flows[s] for s in SEGMENTS},
    }
    state["history"].append(snapshot)
    state["history"] = state["history"][-500:]
    return snapshot


def add_water(state: dict[str, Any], liters: float) -> None:
    before = state["reservoir_volume_l"]
    state["reservoir_volume_l"] = float(np.clip(
        before + liters, 0.0, state["reservoir_capacity_l"]
    ))
    log_event(
        state,
        "Reservoir fill",
        f"Operator added {state['reservoir_volume_l'] - before:.1f} L to the reservoir.",
    )


def remove_water(state: dict[str, Any], liters: float) -> None:
    before = state["reservoir_volume_l"]
    state["reservoir_volume_l"] = float(np.clip(
        before - liters, 0.0, state["reservoir_capacity_l"]
    ))
    log_event(
        state,
        "Reservoir drain",
        f"Operator removed {before - state['reservoir_volume_l']:.1f} L from the reservoir.",
    )


def create_leak(state: dict[str, Any], segment: str, severity_mm: float) -> None:
    state["leaks"][segment] = {
        "active": True,
        "severity_mm": float(severity_mm),
        "patched": False,
    }
    log_event(
        state,
        f"Spike used on {segment}",
        f"A {severity_mm:.1f} mm equivalent opening was created to simulate a physical leak.",
        "CRITICAL",
    )


def patch_leak(state: dict[str, Any], segment: str) -> None:
    leak = state["leaks"][segment]
    if leak["active"] and not leak["patched"]:
        leak["patched"] = True
        log_event(
            state,
            f"Hammer patch applied to {segment}",
            "The simulated opening was sealed. The AI should observe pressure recovery and falling leak flow.",
            "INFO",
        )
    else:
        log_event(
            state,
            f"Patch attempted on {segment}",
            "No open leak was present on this segment.",
            "WARNING",
        )
