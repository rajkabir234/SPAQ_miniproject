import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.svm import SVR  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_CSV = os.path.join(BASE_DIR, "spaq_features.csv")


def train_and_evaluate():
    print("=" * 60)
    print("STEP 15: SVR MODEL TRAINING & EVALUATION")
    print("=" * 60)

    if not os.path.exists(FEATURE_CSV):
        raise FileNotFoundError(f"Feature dataset not found at: {FEATURE_CSV}\nRun feature_extractor.py first.")

    df = pd.read_csv(FEATURE_CSV)

    # Separate Features (X) and Target (y = MOS)
    X = df.iloc[:, 1:-1].values
    y = df["MOS"].values

    print(f"Loaded dataset: {X.shape[0]} samples with {X.shape[1]} low-level features.")

    # 1. Clean Inf and NaN values
    # Replace +inf and -inf with NaN first
    X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"Detected {nan_count} non-finite/NaN values across features. Imputing with column medians...")

    # Impute NaNs using column median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Scale Features (FIT SCALER ONLY ON TRAIN DATA TO PREVENT DATA LEAKAGE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training SVR with RBF Kernel...")
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = svr.predict(X_test_scaled)

    # Calculate required performance metrics
    plcc, _ = pearsonr(y_test, y_pred)
    srcc, _ = spearmanr(y_test, y_pred)
    krcc, _ = kendalltau(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "=" * 60)
    print("PERFORMANCE EVALUATION METRICS")
    print("=" * 60)
    print(f"PLCC (Pearson Linear Correlation Coefficient)  : {plcc:.4f}")
    print(f"SRCC (Spearman Rank Order Correlation Coeff) : {srcc:.4f}")
    print(f"KRCC (Kendall Rank Correlation Coefficient)  : {krcc:.4f}")
    print(f"RMSE (Root Mean Squared Error)               : {rmse:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()