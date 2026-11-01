"""Download the Fama–French daily market data and build the daily log-returns.

Reads:     config.toml ([data] source and window, [paths])
Produces:  data/raw/F-F_Research_Data_Factors_daily.csv   raw snapshot, never overwritten
           data/returns.csv                                 daily log-returns (columns: date, r)
Run:       python scripts/01_download_data.py
"""

# Imports
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))  # make code/src importable
from src.config import load_config  # noqa: E402

# Constants (read from config.toml)
CFG = load_config()
DATA, PATHS = CFG["data"], CFG["paths"]
RAW_FILE = PATHS["raw_dir"] / DATA["raw_file"]


# %% 1. Download the raw file
# DECISION: download only if no snapshot exists, so later revisions of the source never change our results.
if not RAW_FILE.exists():
    PATHS["raw_dir"].mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA["url"]) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extract(DATA["raw_file"], PATHS["raw_dir"])
print("Source vintage:", RAW_FILE.read_text().splitlines()[0])

# %% 2. Parse the file
# The file has 4 lines of notes, a header row, one row per day (date as YYYYMMDD) and a final copyright line.
raw = pd.read_csv(RAW_FILE, skiprows=4, index_col=0)
raw.index = pd.to_datetime(raw.index.astype(str), format="%Y%m%d", errors="coerce")
raw = raw[raw.index.notna()]  # drops the copyright line, the only row without a date

# %% 3. Build the daily log-returns over the chosen window
# ASSUMPTION: market return = Mkt-RF + RF (the Fama–French definition), given in percent.
# ASSUMPTION: the source rounds returns to 0.01%, negligible next to daily moves of about 1%.
R = (raw["Mkt-RF"] + raw["RF"]) / 100
r = np.log1p(R)[DATA["start"]:DATA["end"]].rename("r").rename_axis("date")
assert r.notna().all() and (r > -1).all(), "missing values (NaN or the -99.99 code) in the window"

# %% 4. Save and print the key numbers
r.to_csv(PATHS["returns"])
print(f"{len(r):,} trading days, {r.index[0]:%Y-%m-%d} to {r.index[-1]:%Y-%m-%d}")
print(f"worst day {100 * r.min():.2f}%, best day {100 * r.max():.2f}%  ->  {PATHS['returns']}")
