# ingestion/file_processor.py
import os
import time
import pandas as pd
import shutil
from ingestion.utils import log

# Define folders
DOWNLOADS = os.path.join("data", "downloads")
PROCESSED = os.path.join("data", "processed")
ERROR = os.path.join("data", "error")

# Ensure folders exist
os.makedirs(DOWNLOADS, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)   
os.makedirs(ERROR, exist_ok=True)


def save_attachment(part, eid):
    """Save email attachment into downloads folder"""
    filename = part.get_filename()
    if not filename:
        log.warning(f"Email {eid} has part with no filename, skipping")
        return None

    filepath = os.path.join(DOWNLOADS, filename)
    with open(filepath, "wb") as f:
        f.write(part.get_payload(decode=True))
    log.info(f"Saved attachment {filename} from email {eid} to {filepath}")
    return filepath


def read_file(path):
    log.info(f"Reading file {path}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        log.warning(f"Unsupported file format: {path}")
        return pd.DataFrame()


def normalize_dates(df, file_path: str | None = None):
    if file_path:
        log.info(f"Normalizing dates for file {file_path}")
    else:
        log.info("Normalizing date columns")

    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].dt.strftime("%Y-%m-%d")
            log.debug(f"Normalized column {col} to YYYY-MM-DD")
    return df


def generate_file_hash(path):
    import hashlib
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(BUF_SIZE):
            sha256.update(chunk)
    fhash = sha256.hexdigest()
    log.debug(f"Generated hash for {path}: {fhash}")
    return fhash


def safe_move(src, dst_folder):
    """Move a file safely into processed/error folder, handle duplicates"""
    fname = os.path.basename(src)
    dst = os.path.join(dst_folder, fname)

    # If file already exists, append timestamp
    if os.path.exists(dst):
        base, ext = os.path.splitext(fname)
        fname = f"{base}_{int(time.time())}{ext}"
        dst = os.path.join(dst_folder, fname)

    shutil.move(src, dst)
    return dst
