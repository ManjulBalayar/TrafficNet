"""
Downloads a small UA-DETRAC subset for evaluation using parallel requests.
Grabs N frames from each target sequence via the Kaggle Python API.

Usage:  python download_detrac.py
"""

import os
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import kaggle

# ── Config ────────────────────────────────────────────────────────────────────
SEQUENCES  = ["MVI_20011", "MVI_20032", "MVI_20061"]
FRAMES     = 500      # frames per sequence  (~80 KB each → ~40 MB/seq)
OUTPUT_DIR = "data/ua-detrac"
OWNER      = "bratjay"
DATASET    = "ua-detrac-orig"
IMG_PREFIX = "DETRAC-Images/DETRAC-Images"
WORKERS    = 16       # parallel download threads

api = kaggle.KaggleApi()
api.authenticate()


def download_one(seq, frame_num, dest_dir):
    """Download a single frame and save it. Returns (frame_num, ok)."""
    filename  = f"img{frame_num:05d}.jpg"
    local     = os.path.join(dest_dir, filename)
    if os.path.exists(local):
        return frame_num, True

    remote = f"{IMG_PREFIX}/{seq}/{filename}"
    try:
        response = api.datasets_download_file(
            OWNER, DATASET, remote, _preload_content=False
        )
        data = response.read()

        # Kaggle sometimes wraps single files in a zip
        if data[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                data = z.read(z.namelist()[0])

        with open(local, "wb") as f:
            f.write(data)
        return frame_num, True
    except Exception:
        return frame_num, False


def download_sequence(seq):
    dest = os.path.join(OUTPUT_DIR, "sequences", seq)
    os.makedirs(dest, exist_ok=True)

    # Build list of frames not yet downloaded
    needed = [
        i for i in range(1, FRAMES + 1)
        if not os.path.exists(os.path.join(dest, f"img{i:05d}.jpg"))
    ]
    if not needed:
        print(f"  {seq}: already complete ({FRAMES} frames)")
        return FRAMES, 0

    print(f"  {seq}: downloading {len(needed)} frames with {WORKERS} threads...")
    ok_count = fail_count = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, seq, i, dest): i for i in needed}
        for future in as_completed(futures):
            _, ok = future.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            if (ok_count + fail_count) % 50 == 0:
                print(f"    {ok_count + fail_count}/{len(needed)} done...")

    actual = len([f for f in os.listdir(dest) if f.endswith(".jpg")])
    print(f"  {seq}: {actual} frames saved to {dest}")
    return ok_count, fail_count


def main():
    print(f"Downloading {FRAMES} frames × {len(SEQUENCES)} sequences")
    print(f"Estimated size: ~{FRAMES * 80 * len(SEQUENCES) // 1024} MB\n")

    total_ok = total_fail = 0
    for seq in SEQUENCES:
        print(f"\n── {seq} ──────────────────────────────────")
        ok, fail = download_sequence(seq)
        total_ok   += ok
        total_fail += fail

    print(f"Downloaded : {total_ok} frames")
    print(f"Failed     : {total_fail} frames")
    print(f"Location   : {OUTPUT_DIR}/sequences/")
    if total_fail == 0:
        print("\nReady for evaluation. Run:  python evaluate.py")


if __name__ == "__main__":
    main()
