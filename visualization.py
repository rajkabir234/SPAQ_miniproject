import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.svm import SVR  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_CSV = os.path.join(BASE_DIR, "spaq_features.csv")
OUTPUT_FIG = os.path.join(BASE_DIR, "scatter_plot.png")

def generate_scatter_plot():
    df = pd.read_csv(FEATURE_CSV)
    X = df.iloc[:, 1:-1].values
    y = df["MOS"].values

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

    plt.figure(figsize=(8, 6), dpi=300)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color="teal", edgecolor=None, s=25)
    
    # Ideal fit line
    lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
    plt.plot(lims, lims, "r--", alpha=0.75, zorder=3, label="Ideal Fit (y = x)")

    metrics_str = f"PLCC: {plcc:.4f}\nSRCC: {srcc:.4f}\nKRCC: {krcc:.4f}\nRMSE: {rmse:.4f}"
    plt.gca().text(0.05, 0.92, metrics_str, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.title("SVR Predicted MOS vs. Ground Truth MOS", fontsize=14, fontweight="bold")
    plt.xlabel("Original MOS (Human Annotations)", fontsize=12)
    plt.ylabel("Predicted MOS (SVR Model)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(OUTPUT_FIG)
    print(f"Scatter plot saved to: {OUTPUT_FIG}")

if __name__ == "__main__":
    generate_scatter_plot()

    