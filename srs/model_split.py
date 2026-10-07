import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/processed/zimbabwe_weather_model.csv")
OUTPUT_DIR = Path("data/processed/model_data")

TRAIN_END = "2022-12-31 23:00:00"
VALIDATION_END = "2024-12-31 23:00:00"


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("Zimbabwe Weather Intelligence")
print("Time-Based Model Data Split")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_datetime(df["time"])

print("\nFull dataset shape:", df.shape)
print("Date range:", df["time"].min(), "to", df["time"].max())


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Time-based split
# ============================================================

train = df[df["time"] <= TRAIN_END].copy()

validation = df[
    (df["time"] > TRAIN_END) &
    (df["time"] <= VALIDATION_END)
].copy()

test = df[df["time"] > VALIDATION_END].copy()


# ============================================================
# Save datasets
# ============================================================

train.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test.to_csv(OUTPUT_DIR / "test.csv", index=False)


# ============================================================
# Display results
# ============================================================

print("\nSplit results:")
print("-" * 60)

print("Training:")
print("  Shape:", train.shape)
print("  Date range:", train["time"].min(), "to", train["time"].max())

print("\nValidation:")
print("  Shape:", validation.shape)
print("  Date range:", validation["time"].min(), "to", validation["time"].max())

print("\nTest:")
print("  Shape:", test.shape)
print("  Date range:", test["time"].min(), "to", test["time"].max())


# ============================================================
# Check for missing values
# ============================================================

print("\nMissing values:")
print("  Training:", train.isnull().sum().sum())
print("  Validation:", validation.isnull().sum().sum())
print("  Test:", test.isnull().sum().sum())


print("\nFiles saved to:")
print(OUTPUT_DIR)

print("\nTime-based model split completed successfully.")