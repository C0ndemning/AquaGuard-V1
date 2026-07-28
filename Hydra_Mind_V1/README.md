# HydraMind Digital Twin

A redesigned single-screen Streamlit command center featuring:

- Reduced-order hydraulic physics
- Reservoir fill/drain
- Fluid selection
- Pump command versus actual speed
- House tank fill and demand
- Valve controls always visible on the left
- Spike tool to create a leak
- Hammer tool to patch it
- Water-hammer risk
- Real XGBoost inference using the deployed model
- 205-feature verification
- Local XGBoost contribution values
- Reasoned event log

The simulator is a reduced-order engineering model, not CFD or EPANET. The included
XGBoost model is genuinely executed, but BattLeDIM benchmark performance must not be
presented as proof of performance on the physical four-house prototype.
