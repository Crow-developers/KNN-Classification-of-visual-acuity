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

	#-----------------------------------------------------------
	#   Standardized categorical columns ...
	#-----------------------------------------------------------

	log("\nStanderdized categorical columns ...")
	for col in ["VisionClass"]:
		df_prep[col] = df_prep[col].str.strip().str.lower()
		df_prep[col] = df_prep[col].map({"normal": 0, "weak": 1, "myopia": 2, "hyperopia": 3})
		log(f"{col}: normal-->0, weak-->1, myopia-->2, hyperopia-->3")

run()