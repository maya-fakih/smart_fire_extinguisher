"""
train_model.py
==============
Generates synthetic training data using REAL observed temperature ranges
from test logs and the AMG8833 screenshot (hotspot 28.3°C, room 20-23°C).

Candle/lighter = what we demo with. AMG8833 reads modestly:
  Room baseline:  20-23°C
  Candle nearby:  27-32°C peak (danger 2-3)
  Larger flame:   32-45°C peak (danger 4-5)
  (NOT 60-80°C — that would be direct contact with the sensor)

Run ONCE before main.py:
    cd smart_fire_extinguisher/demo
    python train_model.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

SEED = 42
N_SAMPLES = 600
rng = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).parent

FEATURE_KEYS = sorted([
    "smoke_latest", "smoke_avg", "smoke_variance", "smoke_velocity", "smoke_acceleration",
    "heat_grid_latest", "heat_grid_avg", "heat_grid_variance", "heat_grid_velocity",
    "heat_grid_acceleration", "fire_count", "smoke_count", "cluster_count",
    "fire_union_area", "smoke_union_area", "scene_confidence",
    "composite_label_encoded", "scene_label_encoded",
    "fire_union_area_velocity", "smoke_union_area_velocity",
    "glimpsed_fire", "human_near_fire",
])

# Real observed ranges (from logs + screenshot):
#   Room: 20-23°C | Candle trigger: 27-30°C | Bigger flame: 30-45°C
HEAT_RANGES = {
    1: (20.0, 24.0),   # ambient / no fire
    2: (25.0, 28.5),   # candle detected, marginal (matches screenshot: 28.3°C = danger 2)
    3: (28.0, 33.0),   # candle close / lighter
    4: (32.0, 42.0),   # larger flame
    5: (40.0, 55.0),   # serious fire (rare in demo but model needs to know)
}

SMOKE_RANGES = {
    1: (0,   80),
    2: (50,  200),
    3: (150, 500),
    4: (400, 1500),
    5: (1000, 5000),
}

def _make_row(danger: int) -> dict:
    d = danger
    heat_lo, heat_hi = HEAT_RANGES[d]
    smoke_lo, smoke_hi = SMOKE_RANGES[d]

    heat_base  = rng.uniform(heat_lo, heat_hi)
    smoke_base = rng.uniform(smoke_lo, smoke_hi)

    # Chain stats (5-step window, small realistic variance)
    heat_noise  = rng.normal(0, 0.3)
    smoke_noise = rng.normal(0, smoke_base * 0.08 + 1)

    hl = max(0, heat_base + heat_noise)
    ha = max(0, heat_base * rng.uniform(0.92, 1.02))
    hv = rng.uniform(0, 0.8) * (d / 3)
    hvl = rng.uniform(-0.2, 0.8) * (d / 3)
    hac = rng.uniform(-0.1, 0.3) * (d / 4)

    sl = max(0, smoke_base + smoke_noise)
    sa = max(0, smoke_base * rng.uniform(0.85, 1.05))
    sv = rng.uniform(0, smoke_base * 0.15 + 0.5)
    svl = rng.uniform(-5, 30) * (d / 3)
    sac = rng.uniform(-2, 10) * (d / 4)

    # Vision
    fire_count  = 0 if d == 1 else rng.integers(0, min(d, 3))
    smoke_count = 0 if d <= 2 else rng.integers(0, d - 1)
    cluster_count = max(fire_count, smoke_count)

    fua = 0.0 if fire_count == 0 else float(rng.uniform(0.005, 0.03 * d))
    sua = 0.0 if smoke_count == 0 else float(rng.uniform(0.005, 0.025 * d))
    scene_conf = float(rng.uniform(0.25, 0.45)) if d == 1 else float(rng.uniform(0.25, 0.9))

    if d == 1:   comp = 0
    elif d == 2: comp = rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
    elif d == 3: comp = rng.choice([1, 2],    p=[0.4, 0.6])
    elif d == 4: comp = rng.choice([2, 3],    p=[0.4, 0.6])
    else:        comp = 3

    scene_enc = min(3, max(0, d - 1))
    fua_vel = 0.0 if fua == 0 else float(rng.uniform(0, fua * 0.4))
    sua_vel = 0.0 if sua == 0 else float(rng.uniform(0, sua * 0.3))
    glimpsed  = 1.0 if fire_count > 0 else 0.0
    human_nf  = 1.0 if (d >= 4 and rng.random() > 0.6) else 0.0

    return {
        "smoke_latest": sl, "smoke_avg": sa, "smoke_variance": sv,
        "smoke_velocity": svl, "smoke_acceleration": sac,
        "heat_grid_latest": hl, "heat_grid_avg": ha, "heat_grid_variance": hv,
        "heat_grid_velocity": hvl, "heat_grid_acceleration": hac,
        "fire_count": float(fire_count), "smoke_count": float(smoke_count),
        "cluster_count": float(cluster_count),
        "fire_union_area": fua, "smoke_union_area": sua,
        "scene_confidence": scene_conf,
        "composite_label_encoded": float(comp),
        "scene_label_encoded": float(scene_enc),
        "fire_union_area_velocity": fua_vel,
        "smoke_union_area_velocity": sua_vel,
        "glimpsed_fire": glimpsed, "human_near_fire": human_nf,
        "danger_level": danger,
    }

rows = []
for level in range(1, 6):
    for _ in range(N_SAMPLES // 5):
        rows.append(_make_row(level))

df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
csv_path = OUT_DIR / "fake_data.csv"
df.to_csv(csv_path, index=False)
print(f"[train_model] {len(df)} rows saved → {csv_path}")

X = df[FEATURE_KEYS].values
y = df["danger_level"].values - 1

model = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softmax", num_class=5,
    random_state=SEED, eval_metric="mlogloss",
)
model.fit(X, y)

model_path = OUT_DIR / "model.json"
model.save_model(str(model_path))
print(f"[train_model] Model saved → {model_path}")

# Sanity check: 28.3°C candle reading should be danger 2-3
test_candle = {k: 0.0 for k in FEATURE_KEYS}
test_candle["heat_grid_latest"] = 28.3
test_candle["heat_grid_avg"]    = 25.0
test_candle["heat_grid_variance"] = 2.0
test_candle["smoke_latest"]     = 60.0
test_candle["fire_count"]       = 1.0
test_candle["composite_label_encoded"] = 2.0
test_candle["scene_label_encoded"] = 2.0
test_candle["scene_confidence"] = 0.27
test_candle["glimpsed_fire"]    = 1.0
vals = [test_candle[k] for k in FEATURE_KEYS]
pred = int(model.predict(np.array([vals]))[0]) + 1
print(f"[train_model] Sanity check — 28.3°C candle → predicted danger={pred}  (expect 2-3) {'✓' if pred in (2,3) else '⚠ CHECK'}")
print("[train_model] Done.")
