import os
import cv2
from tqdm import tqdm

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "SPAQ_Dataset")

RAW_IMAGE_FOLDER = os.path.join(DATASET_PATH, "Test Images")
PROCESSED_IMAGE_FOLDER = os.path.join(DATASET_PATH, "processed_images")

# Standard target size (512x512 balances low-level detail and extraction speed)
TARGET_SIZE = (512, 512)


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def preprocess_and_save_images():
    print("=" * 60)
    print("STARTING IMAGE PREPROCESSING (STEP 4 & 5)")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(PROCESSED_IMAGE_FOLDER, exist_ok=True)

    raw_files = [
        f for f in os.listdir(RAW_IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Total raw images to process: {len(raw_files)}")
    print(f"Target resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"Output directory: {PROCESSED_IMAGE_FOLDER}\n")

    processed_count = 0
    error_count = 0

    for img_name in tqdm(raw_files, desc="Preprocessing Images"):
        input_path = os.path.join(RAW_IMAGE_FOLDER, img_name)
        output_path = os.path.join(PROCESSED_IMAGE_FOLDER, img_name)

        # Skip processing if already generated (useful if execution gets interrupted)
        if os.path.exists(output_path):
            processed_count += 1
            continue

        try:
            # 1. Read Image
            img = cv2.imread(input_path)
            if img is None:
                error_count += 1
                continue

            # 2. Resize image using bilinear interpolation
            img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

            # 3. Save preprocessed image to disk with standard JPEG encoding
            cv2.imwrite(output_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError processing {img_name}: {e}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed & saved: {processed_count}/{len(raw_files)}")
    print(f"Errors encountered: {error_count}")


if __name__ == "__main__":
    preprocess_and_save_images()