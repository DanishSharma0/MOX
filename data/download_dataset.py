"""
data/download_dataset.py
─────────────────────────
Attempt to download the UCI Online Retail dataset automatically.

The UCI repository requires a browser-based download for this dataset.
This script checks if the file exists and provides instructions if not.

Usage:
    python data/download_dataset.py
"""
import os
import sys

DATASET_FILENAME = "Online Retail.xlsx"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

UCI_URL = "https://archive.ics.uci.edu/dataset/352/online+retail"

def main():
    if os.path.exists(DATASET_PATH):
        print(f"✅ Dataset already exists at: {DATASET_PATH}")
        return

    print("=" * 60)
    print("UCI Online Retail Dataset — Manual Download Required")
    print("=" * 60)
    print()
    print("Automatic download is not available for this dataset.")
    print("Please download it manually:")
    print()
    print(f"  1. Open: {UCI_URL}")
    print(f"  2. Click the 'Download' button")
    print(f"  3. Extract the ZIP file")
    print(f"  4. Copy 'Online Retail.xlsx' to:")
    print(f"     {DATA_DIR}")
    print()
    print("After placing the file, run:  python main.py")
    print()

    # Try a direct download attempt (may not work due to authentication)
    try:
        import requests
        direct_url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
        print("Attempting automatic download …")
        r = requests.get(direct_url, timeout=30, stream=True)
        if r.status_code == 200:
            import zipfile, io
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for name in z.namelist():
                if name.endswith(".xlsx"):
                    z.extract(name, DATA_DIR)
                    extracted = os.path.join(DATA_DIR, name)
                    if extracted != DATASET_PATH:
                        os.rename(extracted, DATASET_PATH)
                    print(f"✅ Downloaded and extracted to: {DATASET_PATH}")
                    return
        else:
            print(f"Download failed (HTTP {r.status_code}). Please download manually.")
    except Exception as exc:
        print(f"Automatic download failed: {exc}")
        print("Please follow the manual instructions above.")


if __name__ == "__main__":
    main()
