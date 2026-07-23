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

	log("\nDetect & cap outliers (IQR) ...")
	feature_cols = [c for c in numeric_cols if c != "VisionClass"]
	outlier_info = {}
	for col in feature_cols:
		Q1 = df_prep[col].quantile(0.25)
		Q3 = df_prep[col].quantile(0.75)
		IQR = Q3 - Q1
		L = Q1 - 1.5 * IQR
		H = Q3 + 1.5 * IQR
		outlier = ((df_prep[col] < L) | (df_prep[col] > H)).sum()
		if outlier > 0:
			outlier_info[col] = (outlier, L, H)
			df_prep[col] = df_prep[col].clip(lower=L, upper=H)
			log(f'{col}: {outlier} outlier capped --> [{L:.1f}, {H:.1f}]')
		if not outlier_info:
			log("No outliers detected")

	# -----------------------------------------------------------
	#   Remove Duplicates ...
	# -----------------------------------------------------------

	log("\nRemove Duplicates ...")
	n_dup = df_prep.duplicated().sum()
	log(f"\nDuplicates found: {n_dup}")
	if n_dup > 0:
		df_prep.drop_duplicates(inplace=True)
		df_prep.reset_index(drop=True, inplace=True)

	# -----------------------------------------------------------
	#   Feature / target split ...
	# -----------------------------------------------------------

	x = df_prep.drop("VisionClass", axis=1)
	y = df_prep["VisionClass"]

	# -----------------------------------------------------------
	#   Train_Test split (stratified) ...
	# -----------------------------------------------------------

	X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
	log(f"\nTrain/Test split ({int((1-0.3)*100)}% / {int((0.3)*100)}%):")
	log(f"Training: {X_train.shape}")
	log(f"Testing: {X_test.shape}")

	# -----------------------------------------------------------
	#  Balance Training Data (SMOTE)
	# -----------------------------------------------------------

	log("\nBalancing Training Data (SMOTE)...")

	# Distribution before balancing
	hyperopia_before = (y_train == 0).sum()
	myopia_before = (y_train == 1).sum()
	normal_before = (y_train == 2).sum()
	weak_before = (y_train == 3).sum()

	log("Before SMOTE:")
	log(f"   hyperopia : {hyperopia_before}")
	log(f"   myopia : {myopia_before}")
	log(f"   Normal : {normal_before}")
	log(f"   weak : {weak_before}")

	# Create SMOTE object
	smote = SMOTE(sampling_strategy="auto",random_state=42,k_neighbors=5)

	# Balance ONLY the training data
	X_train, y_train = smote.fit_resample(X_train, y_train)

	# Distribution after balancing
	hyperopia_after = (y_train == 0).sum()
	myopia_after = (y_train == 1).sum()
	normal_after = (y_train == 2).sum()
	weak_after = (y_train == 3).sum()

	log("After SMOTE:")
	log(f"   hyperopia : {hyperopia_after}")
	log(f"   myopia : {myopia_after}")
	log(f"   Normal : {normal_after}")
	log(f"   weak : {weak_after}")

	log(f"Training samples after balancing: {X_train.shape[0]}")

	# -----------------------------------------------------------
	# Feature Scaling (StandardScaler)....
	# -----------------------------------------------------------
	log("\nFeature Scaling (StandardScaler)....")
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	log("Scaling applied to train & test")

	# -----------------------------------------------------------
	# save artifacts for downstream stage....
	# -----------------------------------------------------------

	log("\nsaving artifacts....")
	processed_path = os.path.join(REPORTS_DIR, "processed_Data.csv")
	df_prep.to_csv(processed_path,index=False)
	log(f"\nsaved processed Dataset to {processed_path}")

	artifacts = {
		"X_train_scaled": X_train_scaled,
		"X_test_scaled": X_test_scaled,
		"X_train": X_train,
		"X_test":X_test,
		"y_train":y_train,
		"y_test":y_test,
		"feature_names": x.columns.tolist(),
		"scaler":scaler,
		"label_encoder":l,
	}
	for name ,obj in artifacts.items():
		path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
		joblib.dump(obj, path)
		log(f"{name}.pkl")

run()