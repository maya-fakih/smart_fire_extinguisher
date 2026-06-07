"""
demo/train_model.py
===================
Generates a synthetic CSV matching the EXACT schema of think_database.export_csv()
(i.e., SELECT * FROM think_schema with validated=TRUE rows), then trains XGBoost
and saves to model_weights/xgboost_model.json — the path the real pipeline loads from.

Real column schema (from think_database.py):
  triggered_sensors, sensor_readings, sensor_normalized, composite_label,
  glimpsed_fire, human_near_fire, fire_count, smoke_count, fire_union_area,
  smoke_union_area, cluster_count, scene_label, scene_confidence,
  fire_clusters, raw_detections, frame_image_url,
  true_danger_level, true_action, danger_label, validated

sensor_normalized is a JSON dict:  {"smoke": <float 0-1>, "heat_grid": <float 0-1>}
sensor_readings   is a JSON dict:  {"smoke": <ppm>, "heat_grid": [[...8x8...]]}

Real calibration from logs:
  Ambient room:   heat_grid normalized ~0.31-0.32, max ~24-25°C
  Fire (candle):  heat_grid normalized ~0.34-0.38, max ~27-30°C
  Smoke sensor:   always near 0 normalized in tests (7e-8 ppm)
  heat_grid threshold = 27°C → normalized ~0.34

Run:
    cd smart_fire_extinguisher
    python demo/train_model.py
"""

import json, math, random
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)
random.seed(SEED)

REPO     = Path(__file__).resolve().parents[1]
OUT_CSV  = REPO / "demo" / "fake_data.csv"
OUT_MDL  = REPO / "model_weights" / "xgboost_model.json"
OUT_MDL.parent.mkdir(parents=True, exist_ok=True)

# ── Feature keys (must match think_database.build_feature_vector exactly) ────
SENSOR_LIST  = ["smoke", "heat_grid"]   # from config sensor_list
FEATURE_KEYS = sorted([
    "smoke_latest", "smoke_avg", "smoke_variance", "smoke_velocity", "smoke_acceleration",
    "heat_grid_latest", "heat_grid_avg", "heat_grid_variance", "heat_grid_velocity",
    "heat_grid_acceleration",
    "fire_count", "smoke_count", "cluster_count",
    "fire_union_area", "smoke_union_area", "scene_confidence",
    "composite_label_encoded", "scene_label_encoded",
    "fire_union_area_velocity", "smoke_union_area_velocity",
    "glimpsed_fire", "human_near_fire",
])

# ── Real observed normalized ranges (from logs) ───────────────────────────────
# sensor_normalized["heat_grid"] = max(grid) normalized by sensor raw_max
# Logs: ambient=0.31-0.32, candle trigger=0.34-0.38
# sensor_normalized["smoke"] ≈ 0.0 in all test runs
HEAT_NORM = {1:(0.28,0.33), 2:(0.33,0.36), 3:(0.35,0.39), 4:(0.38,0.45), 5:(0.43,0.55)}
SMOKE_NORM= {1:(0.0,0.01),  2:(0.0,0.03),  3:(0.01,0.08), 4:(0.05,0.20), 5:(0.15,0.60)}

DANGER_LABELS = {1:"MINIMAL", 2:"LOW", 3:"MODERATE", 4:"HIGH", 5:"CRITICAL"}
DANGER_ACTIONS= {1:"monitor", 2:"monitor", 3:"alert", 4:"alert,suppress", 5:"alert,suppress,evacuate"}

def _fake_8x8(heat_norm):
    """Generate a plausible 8x8 grid with one hot spot matching the normalized value."""
    # raw_max from config is ~80°C; work backwards: temp = norm * 80
    hot_temp  = heat_norm * 80.0
    base_temp = rng.uniform(22.0, 24.5)
    grid = [[round(float(rng.uniform(base_temp-1, base_temp+1)), 2) for _ in range(8)] for _ in range(8)]
    hr, hc = rng.integers(0,8), rng.integers(0,8)
    grid[hr][hc] = round(hot_temp, 2)
    return grid

def _make_row(danger):
    hn_lo, hn_hi = HEAT_NORM[danger]
    sn_lo, sn_hi = SMOKE_NORM[danger]
    hn = float(rng.uniform(hn_lo, hn_hi))
    sn = float(rng.uniform(sn_lo, sn_hi))

    # ── Realistic NaN patterns from real logs ─────────────────────────────────
    # Real pipeline: sensor features are NaN when chain has < 2 points,
    # or when sense_queue and see_queue timestamps don't align (gap > 500ms).
    # From logs: this happens ~20-30% of frames. Train with this pattern so
    # the model doesn't catastrophically default to danger=5 on NaN inputs.
    sensor_nan = rng.random() < 0.25   # 25% of rows have all-NaN sensor chain
    # Vision features are NaN when camera not triggered or no detections
    vision_nan = (danger == 1 and rng.random() < 0.6) or \
                 (danger == 2 and rng.random() < 0.3)

    # sensor_normalized JSON — use -1 as sentinel for NaN (JSON can't store NaN)
    if sensor_nan:
        sensor_normalized = {"smoke": -1, "heat_grid": -1}
        triggered = []
    else:
        sensor_normalized = {"smoke": round(sn,6), "heat_grid": round(hn,6)}
    sensor_readings = {"smoke": round(sn*10000,4) if not sensor_nan else 0,
                       "heat_grid": _fake_8x8(hn) if not sensor_nan else [[24.0]*8]*8}

    triggered = []
    if hn > 0.337: triggered.append("heat_grid")
    if sn > 0.03:  triggered.append("smoke")

    # vision
    fire_count  = 0 if danger==1 else int(rng.integers(0, min(danger,3)))
    smoke_count = 0 if danger<=2 else int(rng.integers(0, danger-1))
    cluster_count = max(fire_count, smoke_count)
    fua = 0.0 if fire_count==0 else float(rng.uniform(0.005, 0.03*danger))
    sua = 0.0 if smoke_count==0 else float(rng.uniform(0.005, 0.025*danger))
    scene_conf = float(rng.uniform(0.25,0.45)) if danger==1 else float(rng.uniform(0.3,0.9))

    if danger==1:   comp="none"
    elif danger==2: comp=random.choice(["none","smoke","fire"])
    elif danger==3: comp=random.choice(["smoke","fire"])
    elif danger==4: comp=random.choice(["fire","fire_smoke"])
    else:           comp="fire_smoke"

    scene = "fire" if fire_count>0 else ("smoky" if smoke_count>0 else ("hazy" if sn>0.03 else "clear"))

    comp_enc  = {"none":0,"smoke":1,"fire":2,"fire_smoke":3}[comp]
    scene_enc = {"clear":0,"hazy":1,"smoky":2,"fire":3}[scene]

    glimpsed_fire   = 1 if fire_count>0 else 0
    human_near_fire = 1 if (danger>=4 and rng.random()>0.6) else 0

    return {
        # DB columns (for the CSV the real pipeline can load)
        "triggered_sensors":  json.dumps(triggered),
        "sensor_readings":    json.dumps(sensor_readings),
        "sensor_normalized":  json.dumps(sensor_normalized),
        "composite_label":    comp,
        "glimpsed_fire":      glimpsed_fire,
        "human_near_fire":    human_near_fire,
        "fire_count":         fire_count,
        "smoke_count":        smoke_count,
        "fire_union_area":    round(fua,6),
        "smoke_union_area":   round(sua,6),
        "cluster_count":      cluster_count,
        "scene_label":        scene,
        "scene_confidence":   round(scene_conf,4),
        "fire_clusters":      json.dumps([]),
        "raw_detections":     json.dumps([]),
        "frame_image_url":    None,
        "true_danger_level":  danger,
        "true_action":        DANGER_ACTIONS[danger],
        "danger_label":       DANGER_LABELS[danger],
        "danger_level":       danger,
        "recommended_action": DANGER_ACTIONS[danger],
        "validated":          True,
    }

# ── Generate ──────────────────────────────────────────────────────────────────
N = 600
rows = []
for level in range(1, 6):
    for _ in range(N//5):
        rows.append(_make_row(level))

df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"[train] {len(df)} rows → {OUT_CSV}")

# ── Build feature matrix (same logic as think_database.build_feature_vector) ─
def _norm(row, sensor):
    d = json.loads(row["sensor_normalized"])
    return d.get(sensor, float("nan"))

def _feat(df):
    X = []
    for _, row in df.iterrows():
        sn = _norm(row, "smoke")
        hn = _norm(row, "heat_grid")
        # Propagate NaN: if sensor_normalized has nan sentinel (-1), use np.nan
        sn = np.nan if sn < 0 else sn
        hn = np.nan if hn < 0 else hn
        fc = row["fire_count"]; sc2 = row["smoke_count"]; cc = row["cluster_count"]
        fua = row["fire_union_area"]; sua = row["smoke_union_area"]
        comp_enc  = {"none":0,"smoke":1,"fire":2,"fire_smoke":3}.get(row["composite_label"],0)
        scene_enc_map = {"clear":0,"hazy":1,"smoky":2,"fire":3}
        scene_enc = scene_enc_map.get(row["scene_label"], np.nan)
        feat = {
            "smoke_latest":sn, "smoke_avg":sn, "smoke_variance":0.0 if not np.isnan(sn) else np.nan,
            "smoke_velocity":0.0 if not np.isnan(sn) else np.nan,
            "smoke_acceleration":0.0 if not np.isnan(sn) else np.nan,
            "heat_grid_latest":hn, "heat_grid_avg":hn,
            "heat_grid_variance":0.0 if not np.isnan(hn) else np.nan,
            "heat_grid_velocity":0.0 if not np.isnan(hn) else np.nan,
            "heat_grid_acceleration":0.0 if not np.isnan(hn) else np.nan,
            "fire_count":float(fc), "smoke_count":float(sc2), "cluster_count":float(cc),
            "fire_union_area":fua, "smoke_union_area":sua,
            "scene_confidence":row["scene_confidence"],
            "composite_label_encoded":float(comp_enc),
            "scene_label_encoded":scene_enc,
            "fire_union_area_velocity":0.0, "smoke_union_area_velocity":0.0,
            "glimpsed_fire":float(row["glimpsed_fire"]),
            "human_near_fire":float(row["human_near_fire"]),
        }
        X.append([feat[k] for k in FEATURE_KEYS])
    return np.array(X, dtype=float)

X = _feat(df)
y = df["true_danger_level"].values - 1   # 0-4

model = xgb.XGBClassifier(
    n_estimators=120, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softmax", num_class=5,
    random_state=SEED, eval_metric="mlogloss",
)
model.fit(X, y)
model.save_model(str(OUT_MDL))
print(f"[train] Model → {OUT_MDL}")

# ── Sanity checks ─────────────────────────────────────────────────────────────
# 1. Candle reading (what real logs show when fire present)
test = {k:0.0 for k in FEATURE_KEYS}
test["heat_grid_latest"] = 0.341; test["heat_grid_avg"] = 0.338
test["fire_count"] = 1.0; test["composite_label_encoded"] = 2.0
test["scene_label_encoded"] = 3.0; test["scene_confidence"] = 0.5
test["glimpsed_fire"] = 1.0
pred = int(model.predict(np.array([[test[k] for k in FEATURE_KEYS]]))[0]) + 1
print(f"[train] Sanity 1 — candle (norm=0.341, fire detected) → danger={pred}  {'✓' if pred in (2,3,4) else '⚠'}")

# 2. ALL NaN (what real logs show constantly) — must NOT be 5
all_nan = {k: np.nan for k in FEATURE_KEYS}
all_nan["fire_count"] = 0.0; all_nan["smoke_count"] = 0.0
all_nan["cluster_count"] = 0.0; all_nan["fire_union_area"] = 0.0
all_nan["smoke_union_area"] = 0.0; all_nan["scene_confidence"] = 0.0
all_nan["glimpsed_fire"] = 0.0; all_nan["human_near_fire"] = 0.0
all_nan["composite_label_encoded"] = 0.0
pred_nan = int(model.predict(np.array([[all_nan[k] for k in FEATURE_KEYS]]))[0]) + 1
print(f"[train] Sanity 2 — all NaN sensors, no detections → danger={pred_nan}  {'✓' if pred_nan <= 2 else '⚠ STILL BAD — retrain needed'}")
print("[train] Done.")

# ── Evaluation plots ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

DANGER_NAMES = ["MINIMAL\n(1)", "LOW\n(2)", "MODERATE\n(3)", "HIGH\n(4)", "CRITICAL\n(5)"]
COLORS       = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]
PLOT_DIR     = REPO / "demo"

print("[eval] Running cross-validated evaluation …")

# Cross-val predictions (5-fold) for unbiased metrics
y_pred_cv = cross_val_predict(model, X, y, cv=5)
y_pred_cv_labels = y_pred_cv + 1   # back to 1-5
y_true_labels    = y + 1

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true_labels, y_pred_cv_labels, labels=[1,2,3,4,5]
)
report = classification_report(y_true_labels, y_pred_cv_labels,
                                target_names=[f"L{i}" for i in range(1,6)])
print("[eval] Classification report:\n", report)

# ── Figure 1: Precision / Recall / F1 bar chart ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("XGBoost — Per-Class Metrics (5-Fold CV)", fontsize=14, fontweight="bold")

for ax, values, title in zip(axes,
                              [precision, recall, f1],
                              ["Precision", "Recall", "F1-Score"]):
    bars = ax.bar(DANGER_NAMES, values, color=COLORS, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("Danger Level")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
p1 = PLOT_DIR / "eval_precision_recall_f1.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"[eval] Saved → {p1}")

# ── Figure 2: Confusion matrix ────────────────────────────────────────────────
cm = confusion_matrix(y_true_labels, y_pred_cv_labels, labels=[1,2,3,4,5])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("XGBoost — Confusion Matrices (5-Fold CV)", fontsize=14, fontweight="bold")

for ax, data, title, fmt in [
    (ax1, cm,      "Raw Counts",   "d"),
    (ax2, cm_norm, "Normalised",   ".2f"),
]:
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"L{i}" for i in range(1,6)])
    ax.set_yticklabels([f"L{i}" for i in range(1,6)])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    for i in range(5):
        for j in range(5):
            val = data[i,j]
            txt = f"{val:{fmt}}"
            color = "white" if (data[i,j] / data.max()) > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
p2 = PLOT_DIR / "eval_confusion_matrix.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[eval] Saved → {p2}")

# ── Figure 3: ROC curves (one-vs-rest) ────────────────────────────────────────
y_bin   = label_binarize(y_true_labels, classes=[1,2,3,4,5])
y_prob  = cross_val_predict(model, X, y, cv=5, method="predict_proba")

fig, ax = plt.subplots(figsize=(8, 6))
for i, (name, color) in enumerate(zip(DANGER_NAMES, COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name.replace(chr(10),' ')} (AUC={roc_auc:.2f})")

ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5, label="Random")
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — One-vs-Rest (5-Fold CV)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
p3 = PLOT_DIR / "eval_roc_curves.png"
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"[eval] Saved → {p3}")

# ── Figure 4: Feature importance ──────────────────────────────────────────────
importances = model.feature_importances_
idx = np.argsort(importances)[::-1]
sorted_feats = [FEATURE_KEYS[i] for i in idx]
sorted_imps  = importances[idx]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(sorted_feats[::-1], sorted_imps[::-1],
               color=[COLORS[min(int(v*20), 4)] for v in sorted_imps[::-1]],
               edgecolor="white")
ax.set_xlabel("Importance Score", fontsize=11)
ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, sorted_imps[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
plt.tight_layout()
p4 = PLOT_DIR / "eval_feature_importance.png"
plt.savefig(p4, dpi=150, bbox_inches="tight")
plt.close()
print(f"[eval] Saved → {p4}")

print(f"\n[eval] All plots saved to {PLOT_DIR}/")
print(f"       eval_precision_recall_f1.png")
print(f"       eval_confusion_matrix.png")
print(f"       eval_roc_curves.png")
print(f"       eval_feature_importance.png")

# ── Figure 5: Train/Test split comparison — 70/30, 60/40, 50/50 ───────────────
from sklearn.model_selection import train_test_split

SPLITS = [
    ("70/30", 0.30),
    ("60/40", 0.40),
    ("50/50", 0.50),
]
METRICS = ["Precision", "Recall", "F1-Score", "Accuracy"]
split_results = {}

print("[eval] Running train/test split comparisons …")
for label, test_size in SPLITS:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    m = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softmax", num_class=5,
        random_state=SEED, eval_metric="mlogloss",
    )
    m.fit(X_tr, y_tr)
    y_hat = m.predict(X_te)

    p, r, f, _ = precision_recall_fscore_support(y_te, y_hat, average="weighted")
    acc = float((y_hat == y_te).mean())
    split_results[label] = {"Precision": p, "Recall": r, "F1-Score": f, "Accuracy": acc}
    print(f"  [{label}]  P={p:.3f}  R={r:.3f}  F1={f:.3f}  Acc={acc:.3f}  "
          f"(train={len(X_tr)}, test={len(X_te)})")

# ── Grouped bar chart ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
x      = np.arange(len(METRICS))
width  = 0.22
split_colors = ["#3498db", "#e67e22", "#2ecc71"]

for i, (label, color) in enumerate(zip(split_results, split_colors)):
    vals = [split_results[label][m] for m in METRICS]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=f"Split {label}",
                  color=color, edgecolor="white", alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("XGBoost — Performance Across Train/Test Splits", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Add train size annotation
for i, (label, test_size) in enumerate(SPLITS):
    n_train = int(len(X) * (1 - test_size))
    n_test  = len(X) - n_train
    ax.annotate(f"train={n_train}\ntest={n_test}",
                xy=(i * width - width, 0.02),
                fontsize=7.5, color=split_colors[i], alpha=0.8,
                ha="center")

plt.tight_layout()
p5 = PLOT_DIR / "eval_split_comparison.png"
plt.savefig(p5, dpi=150, bbox_inches="tight")
plt.close()
print(f"[eval] Saved → {p5}")

# ── Figure 6: Per-class F1 across splits (line chart) ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
split_colors_line = ["#3498db", "#e67e22", "#2ecc71"]
markers = ["o", "s", "^"]

for (label, test_size), color, marker in zip(SPLITS, split_colors_line, markers):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    m = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softmax", num_class=5,
        random_state=SEED, eval_metric="mlogloss",
    )
    m.fit(X_tr, y_tr)
    y_hat = m.predict(X_te)
    _, _, f1_per, _ = precision_recall_fscore_support(y_te, y_hat, labels=[0,1,2,3,4])
    ax.plot([1,2,3,4,5], f1_per, color=color, marker=marker,
            lw=2, markersize=8, label=f"Split {label}")
    for lvl, val in zip([1,2,3,4,5], f1_per):
        ax.annotate(f"{val:.2f}", (lvl, val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=color)

ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels([f"L{i}\n{n}" for i, n in enumerate(
    ["MINIMAL","LOW","MODERATE","HIGH","CRITICAL"], 1)], fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1-Score", fontsize=11)
ax.set_title("Per-Class F1 Across Train/Test Splits", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
p6 = PLOT_DIR / "eval_split_f1_per_class.png"
plt.savefig(p6, dpi=150, bbox_inches="tight")
plt.close()
print(f"[eval] Saved → {p6}")

print(f"\n[eval] All plots saved to {PLOT_DIR}/")
print(f"       eval_precision_recall_f1.png  — cross-val per-class metrics")
print(f"       eval_confusion_matrix.png     — raw + normalised confusion")
print(f"       eval_roc_curves.png           — ROC one-vs-rest")
print(f"       eval_feature_importance.png   — feature ranking")
print(f"       eval_split_comparison.png     — 70/30 vs 60/40 vs 50/50 overall")
print(f"       eval_split_f1_per_class.png   — per-class F1 across splits")