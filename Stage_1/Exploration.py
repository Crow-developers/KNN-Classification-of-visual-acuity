import os
import sys
import pandas as pd
# ------------------- path ---------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(file)))
DATASET_PATH = os.path.join(BASE_DIR, "vision_dataset_messy.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COL = "VisionClass"
CATEGORICAL_COLS = ["FamilyHistory", "VisionClass"]

def run():
  print("=" * 70)
  print("     STAGE 1 : Data load & Exploration")
  print("=" * 70)

# ----------------------- Load -----------------------
  df = pd.read_csv(DATASET_PATH)
  print(f"\nDataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

#---------------------- Overview ---------------------
  print("\n ----- top 5 Rows -----")
  print(df.head())
  print("\n----- Dtype -----")
  print(df.dtypes)
  print("\n---- statistical summary ----")
  print(df.describe())

# --------------------- Missing Values ----------------------
  missing = df.isnull().sum()
  missing_pct = (missing / len(df) * 100).round(2)
  missing_df = pd.DataFrame({"count":missing, "percentage(%)":missing_pct})
  print("\n------ Missing Values ------")
  if missing.sum() > 0:
    print(missing_df)
  else:
    print("\nNo Missing Values")

#---------------------- Duplicates -------------------------
  n_dup = df.duplicated().sum()
  print(f"\n----- Duplicate Rows:{n_dup}")

#--------------------- Label Distribution -------------------
  print("\n------- Label Distribution -------")
  label_count = df[LABEL_COL].value_counts()
  for l, c in label_count.items():
    print(f"    {l:10s}: {c:4d} ({c / len(df) * 100:.1f}%)")

#-------------------- Unique values in categorical columns ---------
  print("\n------ Unique values in categorical columns ------")
  for col in CATEGORICAL_COLS:
    print(f"    {col}: {df[col].unique().tolist()}")

#--------------- Save Report ---------------------
  report_path = os.path.join(OUTPUT_DIR,"01_data_Exploration.txt")
  with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("     STAGE 1 : Data load & Exploration\n")
    f.write("=" * 70+"\n")
    f.write(f"\nDataset loaded: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    f.write("\n ----- top 5 Rows -----\n")
    f.write(f"{df.head().to_string()}\n")
    f.write("\n----- Dtype -----\n")
    f.write(f"{df.dtypes}\n")
    f.write("\n---- statistical summary ----\n")
    f.write(f"{df.describe().to_string()}\n")
    f.write("\n------ Missing Values ------\n")
    if missing.sum() > 0:
      f.write(f"{missing_df}\n")
    else:
      f.write("\nNo Missing Values\n")

    # ---------------------- Duplicates -------------------------
    f.write(f"\n----- Duplicate Rows:{n_dup}\n")

    # --------------------- Label Distribution -------------------
    f.write("\n------- Label Distribution -------\n")
    for l, c in label_count.items():
      f.write(f"    {l:10s}: {c:4d} ({c / len(df) * 100:.1f}%)\n")

    # -------------------- Unique values in categorical columns ---------
    f.write("\n------ Unique values in categorical columns ------\n")
    for col in CATEGORICAL_COLS:
      f.write(f"    {col}: {df[col].unique().tolist()}\n")
  print(f"\n\nReport saved to: {report_path}")

run()
