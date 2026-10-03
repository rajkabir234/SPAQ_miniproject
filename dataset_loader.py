import os
import pandas as pd

# ============================================================
# SPAQ DATASET CONFIGURATION
# ============================================================

# Resolve paths dynamically relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "SPAQ_Dataset")

EXCEL_FILE = os.path.join(DATASET_PATH, "MOS and Image attribute scores.xlsx")
IMAGE_FOLDER = os.path.join(DATASET_PATH, "Test Images")


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    print("=" * 60)
    print("LOADING SPAQ DATASET METADATA")
    print("=" * 60)

    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(f"Excel file not found at:\n{EXCEL_FILE}")

    # Explicitly specify openpyxl engine to avoid excel engine errors
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")

    print(f"\nMetadata rows: {len(df)}")
    print(f"Metadata columns: {len(df.columns)}")
    print("\nColumns detected:")
    print(list(df.columns))

    return df


# ============================================================
# CHECK IMAGE FILES
# ============================================================

def check_images():
    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(f"Image folder not found at:\n{IMAGE_FOLDER}")

    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(valid_extensions)
    ]

    print("\nImage files found in folder:", len(image_files))
    return image_files


# ============================================================
# VALIDATE METADATA <-> IMAGE MAPPING
# ============================================================

def validate_mapping(df, image_files):
    print("\n" + "=" * 60)
    print("CHECKING IMAGE <-> MOS MAPPING")
    print("=" * 60)

    # Detect the correct filename column dynamically (handles slight naming variations)
    image_col = None
    for col in df.columns:
        if col.strip().lower() in ["image name", "image_name", "filename", "image"]:
            image_col = col
            break

    if image_col is None:
        # Fallback to first column if no standard name matched
        image_col = df.columns[0]

    print(f"Using column '{image_col}' for image filename matching.")

    # Convert both to string sets for consistent comparison
    metadata_images = set(df[image_col].astype(str).str.strip())
    actual_images = set(img.strip() for img in image_files)

    missing_images = metadata_images - actual_images
    extra_images = actual_images - metadata_images

    print(f"\nImages in Excel but missing from folder: {len(missing_images)}")
    print(f"Images in folder but missing from Excel: {len(extra_images)}")

    if missing_images:
        print("\nSample missing images:", list(missing_images)[:5])
    if extra_images:
        print("\nSample extra images:", list(extra_images)[:5])

    if len(missing_images) == 0 and len(extra_images) == 0:
        print("\n✓ Image mapping is PERFECT!")
        print("✓ Every image strictly matches the metadata records.")


# ============================================================
# DISPLAY SAMPLE
# ============================================================

def display_sample(df):
    print("\n" + "=" * 60)
    print("FIRST 10 DATASET RECORDS")
    print("=" * 60)

    # Display available key columns
    cols_to_show = [c for c in df.columns if c.strip().lower() in [
        "image name", "mos", "brightness", "colorfulness", "contrast", "noisiness", "sharpness"
    ]]

    if cols_to_show:
        print(df[cols_to_show].head(10).to_string(index=False))
    else:
        print(df.head(10).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df_data = load_metadata()
    images = check_images()
    validate_mapping(df_data, images)
    display_sample(df_data)

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETED SUCCESSFULLY")
    print("=" * 60)