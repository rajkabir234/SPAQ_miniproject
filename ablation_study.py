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

def evaluate_subset(X, y, name):
    X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)

    plcc, _ = pearsonr(y_test, y_pred)
    srcc, _ = spearmanr(y_test, y_pred)
    krcc, _ = kendalltau(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {"Feature Set": name, "PLCC": plcc, "SRCC": srcc, "KRCC": krcc, "RMSE": rmse}

def run_ablation():
    df = pd.read_csv(FEATURE_CSV)
    y = df["MOS"].values

    # Feature index maps based on header naming conventions
    stat_cols = [c for c in df.columns if c.startswith("f_stat_")]
    grad_cols = [c for c in df.columns if c.startswith("f_grad_")]
    sharp_cols = [c for c in df.columns if c.startswith("f_sharp_")]
    text_cols = [c for c in df.columns if c.startswith("f_text_")]
    color_cols = [c for c in df.columns if c.startswith("f_color_")]
    nss_cols = [c for c in df.columns if c.startswith("f_nss_")]

    results = []

    # 1. Single Domain Evaluations
    results.append(evaluate_subset(df[stat_cols].values, y, "Statistical Only"))
    results.append(evaluate_subset(df[grad_cols].values, y, "Gradient Only"))
    results.append(evaluate_subset(df[sharp_cols].values, y, "Sharpness Only"))
    results.append(evaluate_subset(df[text_cols].values, y, "Texture Only"))
    results.append(evaluate_subset(df[color_cols].values, y, "Color Only"))
    results.append(evaluate_subset(df[nss_cols].values, y, "NSS Only"))

    # 2. Cumulative Fusion Evaluations
    results.append(evaluate_subset(df[stat_cols + grad_cols].values, y, "Stat + Gradient"))
    results.append(evaluate_subset(df[stat_cols + grad_cols + sharp_cols].values, y, "Stat + Grad + Sharp"))
    results.append(evaluate_subset(df[stat_cols + grad_cols + sharp_cols + text_cols].values, y, "+ Texture"))
    results.append(evaluate_subset(df[stat_cols + grad_cols + sharp_cols + text_cols + color_cols].values, y, "+ Color"))
    results.append(evaluate_subset(df.iloc[:, 1:-1].values, y, "Multi-source Fusion (All)"))

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("EXPERIMENT A & B: FEATURE ABLATION & FUSION STUDY")
    print("=" * 60)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_ablation()