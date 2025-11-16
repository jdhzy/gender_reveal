import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULT_DIR = os.path.join(PROJECT_ROOT, "metadata", "results")

overall_path = os.path.join(RESULT_DIR, "overall.csv")
by_gender_path = os.path.join(RESULT_DIR, "by_gender.csv")
by_race_path = os.path.join(RESULT_DIR, "by_race.csv")
by_race_gender_path = os.path.join(RESULT_DIR, "by_race_gender.csv")

# ---------- Load data ----------
overall = pd.read_csv(overall_path)
by_gender = pd.read_csv(by_gender_path)
by_race = pd.read_csv(by_race_path)
race_gender = pd.read_csv(by_race_gender_path)

os.makedirs(RESULT_DIR, exist_ok=True)

# ---------- PLOT 1: OVERALL ACCURACY ----------
plt.figure(figsize=(6, 4))

# Ensure correct ordering
overall_sorted = overall.set_index("condition").loc[["baseline", "normalized"]].reset_index()

plt.bar(overall_sorted["condition"], overall_sorted["acc"])
plt.title("Overall Accuracy: Baseline vs Skin-Normalized")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

out_path = os.path.join(RESULT_DIR, "plot_overall_acc.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print("Saved:", out_path)

# ---------- PLOT 2: GENDER ACCURACY ----------
plt.figure(figsize=(6, 4))

genders = by_gender["gender_name"].tolist()
baseline_g = by_gender["baseline"].tolist()
normalized_g = by_gender["normalized"].tolist()

x = range(len(genders))
width = 0.35

plt.bar([p - width/2 for p in x], baseline_g, width, label="Baseline")
plt.bar([p + width/2 for p in x], normalized_g, width, label="Normalized")

plt.xticks(list(x), genders)
plt.title("Accuracy by Gender\n(Before vs After Skin Normalization)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()

out_path = os.path.join(RESULT_DIR, "plot_gender_acc.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print("Saved:", out_path)

# ---------- PLOT 3: RACE ACCURACY ----------
plt.figure(figsize=(10, 5))

races = by_race["race_name"].tolist()
baseline_r = by_race["baseline"].tolist()
normalized_r = by_race["normalized"].tolist()

x = range(len(races))

plt.bar([p - width/2 for p in x], baseline_r, width, label="Baseline")
plt.bar([p + width/2 for p in x], normalized_r, width, label="Normalized")

plt.xticks(list(x), races, rotation=30, ha="right")
plt.title("Accuracy by Race\n(Before vs After Skin Normalization)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()

out_path = os.path.join(RESULT_DIR, "plot_race_acc.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print("Saved:", out_path)

# ---------- PLOT 4: RACE × GENDER HEATMAP (Δ accuracy) ----------
# race_gender has: race_name, gender_name, baseline, normalized, delta
pivot_delta = race_gender.pivot(index="race_name", columns="gender_name", values="delta")

plt.figure(figsize=(7, 5))

im = plt.imshow(pivot_delta.values, aspect="auto")
plt.colorbar(im, label="Δ Accuracy (normalized - baseline)")

plt.xticks(range(len(pivot_delta.columns)), pivot_delta.columns, rotation=45, ha="right")
plt.yticks(range(len(pivot_delta.index)), pivot_delta.index)

plt.title("Change in Accuracy by Race × Gender\n(Normalized - Baseline)")

out_path = os.path.join(RESULT_DIR, "plot_race_gender_heatmap_delta.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print("Saved:", out_path)