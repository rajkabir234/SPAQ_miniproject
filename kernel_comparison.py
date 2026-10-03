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

def compare_kernels():
    df = pd.read_csv(FEATURE_CSV)
    X = df.iloc[:, 1:-1].values
    y = df["MOS"].values

    X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    results = []

    for kernel in kernels:
        svr = SVR(kernel=kernel, C=10.0, epsilon=0.1)
        svr.fit(X_train_scaled, y_train)
        y_pred = svr.predict(X_test_scaled)

        plcc, _ = pearsonr(y_test, y_pred)
        srcc, _ = spearmanr(y_test, y_pred)
        krcc, _ = kendalltau(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({"Kernel": kernel.upper(), "PLCC": plcc, "SRCC": srcc, "KRCC": krcc, "RMSE": rmse})

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("EXPERIMENT C: SVR KERNEL COMPARISON")
    print("=" * 60)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    compare_kernels()