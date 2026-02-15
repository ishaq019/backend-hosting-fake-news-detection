import os
from pathlib import Path
import nltk

BASE_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = BASE_DIR / "nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

nltk.download("stopwords", download_dir=str(NLTK_DATA_DIR))
print("stopwords downloaded to:", NLTK_DATA_DIR)
