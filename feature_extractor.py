import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy  # type: ignore
from skimage.feature import graycomatrix, graycoprops  # type: ignore
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "SPAQ_Dataset")
PROCESSED_FOLDER = os.path.join(DATASET_PATH, "processed_images")
EXCEL_FILE = os.path.join(DATASET_PATH, "MOS and Image attribute scores.xlsx")
OUTPUT_CSV = os.path.join(BASE_DIR, "spaq_features.csv")


def extract_statistical(gray):
    p = gray.flatten().astype(np.float64)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    
    mean_val = np.mean(p)
    std_val = np.std(p)
    var_val = np.var(p)
    skew_val = float(skew(p))
    kurt_val = float(kurtosis(p))
    entropy_val = float(entropy(hist, base=2))
    rms_contrast = std_val
    min_val = float(np.min(p))
    max_val = float(np.max(p))
    
    # Exactly 9 statistical features
    return [mean_val, std_val, var_val, skew_val, kurt_val, entropy_val, rms_contrast, min_val, max_val]


def extract_gradient(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2).flatten()
    
    # Exactly 6 gradient features
    return [
        np.mean(mag),
        np.std(mag),
        np.var(mag),
        float(skew(mag)),
        float(kurtosis(mag)),
        np.mean(mag**2)
    ]


def extract_sharpness(gray):
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(sx**2 + sy**2)
    edges = cv2.Canny(gray, 100, 200)
    density = np.sum(edges > 0) / edges.size
    
    f = np.fft.fftshift(np.fft.fft2(gray))
    spec = np.abs(f)
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    spec[cy-30:cy+30, cx-30:cx+30] = 0
    
    # Exactly 4 sharpness features
    return [lap_var, tenengrad, density, np.mean(spec)]


def extract_texture(gray):
    gray_q = (gray / 8).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
    
    # Exactly 6 texture features
    return [np.mean(graycoprops(glcm, prop)) for prop in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity', 'ASM']]


def extract_color(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    feats = []
    # 1. RGB channel means & stds (6 features)
    for c in range(3):
        vals = rgb[:, :, c].astype(np.float64)
        feats.extend([np.mean(vals), np.std(vals)])
        
    # 2. HSV channel means & stds (6 features)
    for c in range(3):
        vals = hsv[:, :, c].astype(np.float64)
        feats.extend([np.mean(vals), np.std(vals)])
        
    # 3. Lab lightness mean & std (2 features)
    l_vals = lab[:, :, 0].astype(np.float64)
    feats.extend([np.mean(l_vals), np.std(l_vals)])
    
    # 4. Hasler-Süsstrunk Colorfulness metric (1 feature)
    R, G, B = rgb[:, :, 0].astype(np.float64), rgb[:, :, 1].astype(np.float64), rgb[:, :, 2].astype(np.float64)
    rg, yb = np.abs(R - G), np.abs(0.5 * (R + G) - B)
    colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    feats.append(colorfulness)
    
    # Exactly 15 color features
    return feats


def extract_nss(gray):
    gray_f = gray.astype(np.float64)
    mu = cv2.GaussianBlur(gray_f, (7, 7), 1.166)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray_f**2, (7, 7), 1.166) - mu**2))
    mscn = (gray_f - mu) / (sigma + 1.0)
    flat = mscn.flatten()
    
    # Exactly 6 NSS features
    return [np.mean(flat), np.std(flat), float(skew(flat)), float(kurtosis(flat)), np.var(mu), np.var(sigma)]


def run_extraction():
    print("=" * 60)
    print("STEPS 8-14: FEATURE EXTRACTION (FIXED DIMENSIONS)")
    print("=" * 60)
    
    meta_df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    mos_dict = dict(zip(meta_df["Image name"].astype(str), meta_df["MOS"]))
    
    files = sorted([f for f in os.listdir(PROCESSED_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    headers = (["image_name"] + 
               [f"f_stat_{i+1:02d}" for i in range(9)] + 
               [f"f_grad_{i+1:02d}" for i in range(6)] + 
               [f"f_sharp_{i+1:02d}" for i in range(4)] + 
               [f"f_text_{i+1:02d}" for i in range(6)] + 
               [f"f_color_{i+1:02d}" for i in range(15)] + 
               [f"f_nss_{i+1:02d}" for i in range(6)] + 
               ["MOS"])
    
    rows = []
    for img_name in tqdm(files, desc="Extracting Features"):
        if img_name not in mos_dict:
            continue
        
        bgr = cv2.imread(os.path.join(PROCESSED_FOLDER, img_name))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        f_stat = extract_statistical(gray)      # 9
        f_grad = extract_gradient(gray)         # 6
        f_sharp = extract_sharpness(gray)       # 4
        f_text = extract_texture(gray)          # 6
        f_color = extract_color(bgr)            # 15
        f_nss = extract_nss(gray)               # 6
        
        vec = f_stat + f_grad + f_sharp + f_text + f_color + f_nss
        
        row = [img_name] + vec + [mos_dict[img_name]]
        rows.append(row)
        
    df_out = pd.DataFrame(rows, columns=headers)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved feature matrix to: {OUTPUT_CSV}")
    print(f"Dataset Shape: {df_out.shape} (11,125 records x 46 features + image_name + MOS)")


if __name__ == "__main__":
    run_extraction()