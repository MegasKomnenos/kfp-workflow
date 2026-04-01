from __future__ import annotations

FD_CONFIGS = {
    "FD001": {"n_conditions": 1, "fault_modes": "HPC"},
    "FD002": {"n_conditions": 6, "fault_modes": "HPC"},
    "FD003": {"n_conditions": 1, "fault_modes": "HPC + Fan"},
    "FD004": {"n_conditions": 6, "fault_modes": "HPC + Fan"},
}

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
SENSOR_14 = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
RAW_FEATURE_COLS = OP_COLS + [f"s{i}" for i in range(1, 22)]

LITERATURE = {
    "THGNN_TNNLS_2025": {
        "source": "IEEE TNNLS 2025, Temporal and Heterogeneous Graph Neural Network for Remaining Useful Life Prediction (Table 4)",
        "url": "https://doi.org/10.1109/TNNLS.2025.3592788",
        "metrics": {
            "FD001": {"rmse": 13.15, "score": 285.0},
            "FD002": {"rmse": 13.84, "score": 806.0},
            "FD003": {"rmse": 12.61, "score": 255.0},
            "FD004": {"rmse": 14.65, "score": 1166.0},
        },
    },
    "INF_FUSION_2026": {
        "source": "Information Fusion 2026, Building of transformer-based RUL predictors supported by explainability techniques",
        "url": "https://doi.org/10.1016/j.inffus.2025.103892",
        "reference_model": "MHA-LSTM",
        "metrics": {
            "FD001": {"rmse": 11.43, "score": 209.0},
            "FD002": {"rmse": 13.32, "score": 1058.0},
            "FD003": {"rmse": 11.47, "score": 187.0},
            "FD004": {"rmse": 14.38, "score": 1618.0},
        },
    },
}
