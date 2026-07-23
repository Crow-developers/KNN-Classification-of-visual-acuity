import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# ---------------------------- path -----------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "vision_dataset_messy.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
for d in [REPORTS_DIR, ARTIFACTS_DIR]:
	os.makedirs(d, exist_ok=True)

#------------------------------------------------------------------------

def run():
	print("=" * 70)
	print("		STAGE 2 : Data Preprocessing")
	print("=" * 70)
	df = pd.read_csv(DATASET_PATH)
	df_prep = df.copy()
	reports_lines = []
	def log(msg):
		print(msg)
		reports_lines.append(msg)

	log(f"Original shape: {df.shape}")

	# -----------------------------------------------------------
	#   Encode Target Variable ...
	# -----------------------------------------------------------

	log("\nEncode Target Variable ...")
	l = LabelEncoder()
	df_prep["VisionClass"] = l.fit_transform(df_prep["VisionClass"])
	label_map = dict(zip(l.classes_,l.transform(l.classes_)))
	log(f"Mapping: {label_map}")

	# -----------------------------------------------------------
	#   Handle missing values - Median Technique ...
	# -----------------------------------------------------------

	log("\nHandle missing values - Median Technique ...")
	numeric_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
	imputed_cols = {}
	for c in numeric_cols:
		n_missing = df_prep[c].isnull().sum()
		if n_missing > 0:
			median_val = df_prep[c].median()
			df_prep[c] = df_prep[c].fillna(median_val)
			imputed_cols[c] = (n_missing, median_val)
			log(f" {c}: {n_missing} missing --> median = {median_val}")
	remaining = df_prep.isnull().sum()
	log(f" Remaining missing:\n{remaining}")

	# -----------------------------------------------------------
	#   Detect & cap outliers (IQR) ...
	# -----------------------------------------------------------


run()