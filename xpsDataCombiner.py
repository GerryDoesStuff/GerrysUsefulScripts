import os
import re
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd


# -------------------------
# Config
# -------------------------
OUTPUT_NAME = "aggregate.xlsx"
PEAK_TABLE_SHEET = "Peak Table"

# Excel reading is I/O-heavy; threads tend to be the most reliable default.
# Set USE_THREADS=False if you want multi-process (multi-core) execution.
USE_THREADS = True

# If None, uses os.cpu_count()
MAX_WORKERS = None


# -------------------------
# Sorting
# -------------------------
def numeric_sort_key(path_str):
    """
    Numeric-natural sort for filenames.
    Ensures ...40... comes before ...100....
    """
    name = Path(path_str).name
    parts = re.split(r"(\d+)", name)
    key = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return key


# -------------------------
# Spectral detection / reading
# -------------------------
def find_header_row_and_cols(df_raw):
    """
    Find:
      - header row containing "Binding Energy"
      - energy column index
      - first counts/s column (can be on header row or next units row)
    Returns (header_row, energy_col, intensity_col) or (None, None, None).
    """
    header_row = None
    energy_col = None

    # locate "Binding Energy" header row
    for r in range(len(df_raw)):
        row_vals = df_raw.iloc[r].astype(str).str.lower()
        hits = row_vals.str.contains("binding energy", na=False)
        if hits.any():
            header_row = r
            energy_col = hits.idxmax()
            break

    if header_row is None:
        return None, None, None

    def counts_cols_in_row(r):
        if r < 0 or r >= len(df_raw):
            return []
        vals = df_raw.iloc[r].astype(str).str.lower()
        return [c for c in df_raw.columns if "counts" in vals.get(c, "")]

    # counts can be on header row or units row
    counts_cols = counts_cols_in_row(header_row)
    if not counts_cols:
        counts_cols = counts_cols_in_row(header_row + 1)
    if not counts_cols:
        counts_cols = counts_cols_in_row(header_row + 2)

    if not counts_cols:
        return header_row, energy_col, None

    intensity_col = None
    for c in counts_cols:
        if c > energy_col:
            intensity_col = c
            break
    if intensity_col is None:
        intensity_col = counts_cols[0]

    return header_row, energy_col, intensity_col


def read_spectrum_worker(args):
    """
    Worker-friendly spectrum reader.
    Args: (file_path, sheet_name)
    Returns: (file_path, sheet_name, tidy_df_or_None)
    """
    file_path, sheet_name = args
    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine="openpyxl")
        header_row, energy_col, intensity_col = find_header_row_and_cols(df_raw)

        if header_row is None or energy_col is None or intensity_col is None:
            return file_path, sheet_name, None

        data_start = header_row + 2  # header + units

        energy = df_raw.iloc[data_start:, energy_col]
        intensity = df_raw.iloc[data_start:, intensity_col]

        tidy = pd.DataFrame({
            "Binding Energy (eV)": pd.to_numeric(energy, errors="coerce"),
            "Intensity (Counts/s)": pd.to_numeric(intensity, errors="coerce"),
        }).dropna(subset=["Binding Energy (eV)"])

        if tidy.empty:
            return file_path, sheet_name, None

        return file_path, sheet_name, tidy

    except Exception:
        return file_path, sheet_name, None


def list_sheet_names(file_path):
    try:
        xl = pd.ExcelFile(file_path, engine="openpyxl")
        return xl.sheet_names
    except Exception:
        return []


def detect_spectral_sheets(files):
    """
    Ordered list of spectral sheets:
      1) spectral sheets from first file (in-file order)
      2) additional spectral sheets found elsewhere (sorted)
    """
    print("Detecting spectral sheets...", flush=True)

    first_sheets = list_sheet_names(files[0])
    first_order = []
    extra = set()

    for s in first_sheets:
        if s == PEAK_TABLE_SHEET:
            continue
        _, _, tidy = read_spectrum_worker((files[0], s))
        if tidy is not None:
            first_order.append(s)

    for f in files[1:]:
        for s in list_sheet_names(f):
            if s == PEAK_TABLE_SHEET or s in first_order:
                continue
            _, _, tidy = read_spectrum_worker((f, s))
            if tidy is not None:
                extra.add(s)

    spectral = first_order + sorted(extra)
    print(f"Spectral sheets detected: {spectral}", flush=True)
    return spectral


def get_executor():
    Executor = ThreadPoolExecutor if USE_THREADS else ProcessPoolExecutor
    workers = MAX_WORKERS if MAX_WORKERS is not None else os.cpu_count()
    return Executor, workers


def aggregate_spectral_sheet(sheet, files):
    """
    Read spectra for one sheet in parallel, then outer-merge on BE.
    Column order is enforced to follow numeric-natural file order.
    """
    print(f"[{sheet}] Reading spectra...", flush=True)

    Executor, workers = get_executor()
    tasks = [(f, sheet) for f in files]

    spectra_by_stem = {}

    with Executor(max_workers=workers) as ex:
        futures = [ex.submit(read_spectrum_worker, t) for t in tasks]
        for fut in as_completed(futures):
            fpath, _, tidy = fut.result()
            if tidy is None:
                continue
            stem = Path(fpath).stem
            spectra_by_stem[stem] = tidy.rename(columns={"Intensity (Counts/s)": stem})
            print(f"[{sheet}] Loaded {stem} ({len(tidy)} pts)", flush=True)

    if not spectra_by_stem:
        print(f"[{sheet}] No spectra found. Skipping.", flush=True)
        return None

    # Deterministic stem order based on numeric-natural file order
    stems_sorted = [Path(f).stem for f in files if Path(f).stem in spectra_by_stem]

    print(f"[{sheet}] Merging...", flush=True)
    merged = spectra_by_stem[stems_sorted[0]]
    for stem in stems_sorted[1:]:
        merged = pd.merge(
            merged, spectra_by_stem[stem],
            on="Binding Energy (eV)",
            how="outer",
            sort=True
        )

    merged = merged.sort_values("Binding Energy (eV)", ascending=False)

    # Enforce final column order explicitly
    merged = merged[["Binding Energy (eV)"] + stems_sorted]

    print(f"[{sheet}] Merged shape: {merged.shape}", flush=True)
    return merged


# -------------------------
# Peak Table aggregation (horizontal, flattened columns)
# -------------------------
def read_peak_table_worker(file_path):
    try:
        xl = pd.ExcelFile(file_path, engine="openpyxl")
        if PEAK_TABLE_SHEET not in xl.sheet_names:
            return file_path, None
        df_peak = pd.read_excel(file_path, sheet_name=PEAK_TABLE_SHEET, engine="openpyxl")
        if df_peak.empty:
            return file_path, None
        return file_path, df_peak
    except Exception:
        return file_path, None


def aggregate_peak_tables_horizontally(files):
    """
    Combine Peak Tables side-by-side.
    Columns are flattened to single level:
      "<file_stem> | <original_column>"
    Order follows numeric-natural file order.
    """
    print("[Peak Table] Reading peak tables...", flush=True)

    Executor, workers = get_executor()
    with Executor(max_workers=workers) as ex:
        futures = [ex.submit(read_peak_table_worker, f) for f in files]
        results = [fut.result() for fut in as_completed(futures)]

    # restore numeric-natural order
    results.sort(key=lambda x: numeric_sort_key(x[0]))

    blocks = []
    for fpath, df_peak in results:
        if df_peak is None:
            continue
        stem = Path(fpath).stem
        df_peak = df_peak.reset_index(drop=True).copy()
        df_peak.columns = [f"{stem} | {c}" for c in df_peak.columns]
        blocks.append(df_peak)
        print(f"[Peak Table] Loaded {stem} ({df_peak.shape[0]} rows)", flush=True)

    if not blocks:
        print("[Peak Table] None found. Skipping.", flush=True)
        return None

    peak_agg = pd.concat(blocks, axis=1)
    print(f"[Peak Table] Aggregated shape: {peak_agg.shape}", flush=True)
    return peak_agg


# -------------------------
# Main
# -------------------------
def aggregate_folder(folder=".", output_name=OUTPUT_NAME):
    folder = Path(folder)
    files = glob.glob(str(folder / "*.xlsx"))
    files = [f for f in files if Path(f).name != output_name]
    files = sorted(files, key=numeric_sort_key)

    if not files:
        raise RuntimeError("No .xlsx files found.")

    print(f"Found {len(files)} files:", flush=True)
    for f in files:
        print(f"  - {Path(f).name}", flush=True)

    spectral_sheets = detect_spectral_sheets(files)

    out_path = folder / output_name
    print(f"Writing output: {out_path}", flush=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Spectral sheets
        for sheet in spectral_sheets:
            merged = aggregate_spectral_sheet(sheet, files)
            if merged is not None:
                merged.to_excel(writer, sheet_name=sheet, index=False)

        # Peak Table sheet
        peak_agg = aggregate_peak_tables_horizontally(files)
        if peak_agg is not None:
            peak_agg.to_excel(writer, sheet_name=PEAK_TABLE_SHEET, index=False)

    print("Done.", flush=True)


if __name__ == "__main__":
    aggregate_folder()
