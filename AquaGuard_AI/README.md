# AquaGuard AI — Production-model dashboard

This package is a full Streamlit replacement app that performs real inference with:

- `deployment_model.pkl`
- `deployment_features.pkl`
- `deployment_label_encoder.pkl`
- `deployment_metadata.json`
- the exact 205-feature hydraulic pipeline

## Run

Open PowerShell in this folder, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Real CSV format

A CSV must contain:

- `Timestamp`
- all 33 pressure sensors (`n1` ... `n769`)
- `p227`
- `p235`
- `PUMP_1`

At least 6 chronological rows are required because the model uses rolling and change features.

## Important engineering boundary

The XGBoost model predicts leak location. It does not directly control hardware.
The dashboard's recommendation is a separate safety layer, and autonomous actuation
should remain disabled until physical validation is completed.
