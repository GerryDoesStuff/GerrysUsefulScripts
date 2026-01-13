#!/usr/bin/env python3
"""
Scan all .xlsx files in the script's folder, find sheets with "Scan" in the name,
treat column A as X and each subsequent numeric column as a Y series, then:

- normalize each Y series (default: divide by max |y|)
- lightly smooth (default: Savitzky–Golay, window=11, poly=2)
- find the tallest peak (global maximum after smoothing)
- write results to a new .xlsx where:
    * each scan (i.e., each Y column) becomes its own output column (wide format)
    * Etch Time and Etch Level are preserved as NUMBERS and also stored in their own columns
      in a long-format sheet (per-scan rows)

Output workbook contains:
- PeakLocations   (wide: each scan column = peak X)
- PeakHeights     (wide: each scan column = peak height)
- EtchTime        (wide: each scan column = etch time)
- EtchLevel       (wide: each scan column = etch level)
- PeakSummary     (long: one row per scan with numeric EtchTime/EtchLevel + peaks)
- RunParameters
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Prefer Savitzky–Golay; fallback to simple moving average if SciPy isn't available.
try:
    from scipy.signal import savgol_filter  # type: ignore
    _HAS_SCIPY = True
except Exception:
    savgol_filter = None
    _HAS_SCIPY = False


def excel_col_letter(col_idx_0based: int) -> str:
    """0-based column index -> Excel column letters (A, B, ..., Z, AA, AB, ...)."""
    n = col_idx_0based + 1
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _clean_str_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\ufeff", "", regex=False)  # BOM
        .str.strip()
        .str.lower()
    )


def _interp_nans(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    idx = np.arange(n)
    mask = np.isfinite(y)
    if mask.sum() == 0:
        return y
    if mask.sum() == 1:
        y[~mask] = y[mask][0]
        return y
    y2 = y.copy()
    y2[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    return y2


def _pick_best_metadata_row(df: pd.DataFrame, header_row: int, keyword: str) -> Optional[int]:
    """
    Choose the metadata row (above header_row) where column B contains keyword,
    preferring the row with the most numeric entries in columns C..end.
    """
    if df.shape[1] < 3:
        return None
    col1 = _clean_str_series(df.iloc[:header_row, 1])
    hits = col1[col1.str.contains(keyword, na=False)].index.tolist()
    if not hits:
        return None

    best_row = None
    best_count = -1
    for r in hits:
        numeric_count = pd.to_numeric(df.iloc[r, 2:], errors="coerce").notna().sum()
        if int(numeric_count) > best_count:
            best_count = int(numeric_count)
            best_row = int(r)

    return best_row if (best_row is not None and best_count > 0) else None


def _parse_scan_sheet(df: pd.DataFrame) -> Optional[List[Dict[str, object]]]:
    """
    Parse a 'Scan' sheet into a list of scan dicts:
      {
        scan_col_index, scan_col_letter, scan_id,
        etch_time, etch_level,
        x, y
      }

    Heuristic (matches typical XPS exports):
    - Find a row where col A == 'eV' (case-insensitive, BOM-stripped).
    - X data is the contiguous numeric block under that row in col A.
    - Y scans are numeric columns in that block (excluding col B if empty).
    - Etch Time / Etch Level are read from rows above header where col B contains those keywords.
    """
    if df.shape[0] < 5 or df.shape[1] < 2:
        return None

    col0 = _clean_str_series(df.iloc[:, 0])
    header_matches = col0[col0 == "ev"].index
    if len(header_matches) == 0:
        return None
    header_row = int(header_matches[0])

    # Find contiguous numeric block below header in column A
    x_all = pd.to_numeric(df.iloc[header_row + 1 :, 0], errors="coerce")
    valid = x_all.notna()
    if valid.sum() < 3:
        return None

    first_valid_row = int(valid.idxmax())
    valid_after = valid.loc[first_valid_row:]
    first_non_numeric = valid_after[~valid_after].index
    end_row = int(first_non_numeric[0]) if len(first_non_numeric) > 0 else int(df.shape[0])

    data = df.iloc[first_valid_row:end_row].copy()
    x = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)

    # Y columns: numeric in the first data row
    y_cols: List[int] = []
    for c in range(1, data.shape[1]):
        if pd.notna(pd.to_numeric(data.iat[0, c], errors="coerce")):
            y_cols.append(c)
    if not y_cols:
        return None

    etch_time_row = _pick_best_metadata_row(df, header_row, "etch time")
    etch_level_row = _pick_best_metadata_row(df, header_row, "etch level")

    scans: List[Dict[str, object]] = []
    for c in y_cols:
        y = pd.to_numeric(data.iloc[:, c], errors="coerce").to_numpy(dtype=float)

        etch_time = np.nan
        etch_level = np.nan

        if etch_time_row is not None:
            v = pd.to_numeric(df.iat[etch_time_row, c], errors="coerce")
            if pd.notna(v):
                etch_time = float(v)

        if etch_level_row is not None:
            v = pd.to_numeric(df.iat[etch_level_row, c], errors="coerce")
            if pd.notna(v):
                etch_level = float(v)

        col_letter = excel_col_letter(c)
        scan_id = f"Scan_{col_letter}"

        scans.append(
            {
                "scan_col_index": int(c),
                "scan_col_letter": col_letter,
                "scan_id": scan_id,
                "etch_time": etch_time,
                "etch_level": etch_level,
                "x": x,
                "y": y,
            }
        )

    return scans


def _smooth(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 5:
        return y

    if _HAS_SCIPY:
        win = int(window)
        if win % 2 == 0:
            win -= 1
        win = max(3, win)
        if win > n:
            win = n if (n % 2 == 1) else max(3, n - 1)

        poly = min(int(polyorder), win - 1)
        return savgol_filter(y, window_length=win, polyorder=poly, mode="interp")

    # Fallback: simple moving average (odd window)
    win = int(window)
    if win % 2 == 0:
        win -= 1
    win = max(3, min(win, n if n % 2 == 1 else n - 1))
    k = win // 2
    ypad = np.pad(y, (k, k), mode="edge")
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(ypad, kernel, mode="valid")


def _normalize_and_peak(
    x: np.ndarray,
    y: np.ndarray,
    normalize: str,
    window: int,
    polyorder: int,
) -> Optional[Tuple[float, float]]:
    y = _interp_nans(y)

    if normalize == "max":
        m = np.nanmax(np.abs(y))
        if not np.isfinite(m) or m == 0:
            return None
        y_n = y / m
    elif normalize == "minmax":
        mn = np.nanmin(y)
        mx = np.nanmax(y)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            return None
        y_n = (y - mn) / (mx - mn)
    else:
        raise ValueError("normalize must be 'max' or 'minmax'")

    y_s = _smooth(y_n, window=window, polyorder=polyorder)
    i = int(np.nanargmax(y_s))
    return float(x[i]), float(y_s[i])


def _wide_from_long(
    df_long: pd.DataFrame,
    value_col: str,
    index_cols: List[str],
    scan_col: str = "scan_id",
) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=index_cols)

    wide = (
        df_long.pivot_table(index=index_cols, columns=scan_col, values=value_col, aggfunc="first")
        .reset_index()
    )

    # Order scan columns by their underlying Excel column index if available
    scan_cols = [c for c in wide.columns if c not in index_cols]
    def _scan_sort_key(scan_id: str) -> int:
        # scan_id format "Scan_<LETTER>"
        if not isinstance(scan_id, str) or "_" not in scan_id:
            return 10**9
        letter = scan_id.split("_", 1)[1]
        # Convert Excel letters back to index (A=1...)
        idx = 0
        for ch in letter:
            if "A" <= ch <= "Z":
                idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return idx

    scan_cols_sorted = sorted(scan_cols, key=_scan_sort_key)
    return wide[index_cols + scan_cols_sorted]


def process_folder(
    input_dir: Path,
    output_path: Path,
    normalize: str,
    window: int,
    polyorder: int,
) -> None:
    xlsx_files = sorted(
        p for p in input_dir.glob("*.xlsx")
        if not p.name.startswith("~$") and p.resolve() != output_path.resolve()
    )

    rows: List[Dict[str, object]] = []

    for fp in xlsx_files:
        try:
            xls = pd.ExcelFile(fp, engine="openpyxl")
        except Exception:
            continue

        for sheet in xls.sheet_names:
            if "scan" not in sheet.lower():
                continue

            try:
                df = pd.read_excel(xls, sheet_name=sheet, header=None)
            except Exception:
                continue

            scans = _parse_scan_sheet(df)
            if not scans:
                continue

            for s in scans:
                peak = _normalize_and_peak(
                    x=np.asarray(s["x"], dtype=float),
                    y=np.asarray(s["y"], dtype=float),
                    normalize=normalize,
                    window=window,
                    polyorder=polyorder,
                )
                if peak is None:
                    continue

                peak_x, peak_h = peak

                rows.append(
                    {
                        "file": fp.name,
                        "sheet": sheet,
                        "scan_id": s["scan_id"],
                        "scan_col_letter": s["scan_col_letter"],
                        "scan_col_index": s["scan_col_index"],
                        "etch_time": float(s["etch_time"]) if np.isfinite(s["etch_time"]) else np.nan,
                        "etch_level": float(s["etch_level"]) if np.isfinite(s["etch_level"]) else np.nan,
                        "peak_x": peak_x,
                        "peak_height": peak_h,
                    }
                )

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        # Still write an output file with empty sheets + parameters
        params = pd.DataFrame(
            [{
                "normalize": normalize,
                "savgol_window": window,
                "savgol_polyorder": polyorder,
                "input_dir": str(input_dir),
                "scipy_available": bool(_HAS_SCIPY),
            }]
        )
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer, index=False, sheet_name="PeakLocations")
            pd.DataFrame().to_excel(writer, index=False, sheet_name="PeakHeights")
            pd.DataFrame().to_excel(writer, index=False, sheet_name="EtchTime")
            pd.DataFrame().to_excel(writer, index=False, sheet_name="EtchLevel")
            pd.DataFrame().to_excel(writer, index=False, sheet_name="PeakSummary")
            params.to_excel(writer, index=False, sheet_name="RunParameters")
        return

    # Stable ordering
    df_long = df_long.sort_values(["file", "sheet", "scan_col_index"], kind="stable")

    index_cols = ["file", "sheet"]

    # Wide sheets: each scan gets its own column
    df_peak_x_wide = _wide_from_long(df_long, value_col="peak_x", index_cols=index_cols, scan_col="scan_id")
    df_peak_h_wide = _wide_from_long(df_long, value_col="peak_height", index_cols=index_cols, scan_col="scan_id")
    df_etch_time_wide = _wide_from_long(df_long, value_col="etch_time", index_cols=index_cols, scan_col="scan_id")
    df_etch_level_wide = _wide_from_long(df_long, value_col="etch_level", index_cols=index_cols, scan_col="scan_id")

    # Long summary: EtchTime and EtchLevel are numeric columns
    df_summary = df_long[
        ["file", "sheet", "scan_id", "scan_col_letter", "scan_col_index", "etch_time", "etch_level", "peak_x", "peak_height"]
    ].copy()

    params = pd.DataFrame(
        [{
            "normalize": normalize,
            "savgol_window": window,
            "savgol_polyorder": polyorder,
            "input_dir": str(input_dir),
            "scipy_available": bool(_HAS_SCIPY),
        }]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_peak_x_wide.to_excel(writer, index=False, sheet_name="PeakLocations")
        df_peak_h_wide.to_excel(writer, index=False, sheet_name="PeakHeights")
        df_etch_time_wide.to_excel(writer, index=False, sheet_name="EtchTime")
        df_etch_level_wide.to_excel(writer, index=False, sheet_name="EtchLevel")
        df_summary.to_excel(writer, index=False, sheet_name="PeakSummary")
        params.to_excel(writer, index=False, sheet_name="RunParameters")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder containing .xlsx files. Default: folder where this script is located.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .xlsx path. Default: peak_results.xlsx in the input folder.",
    )
    parser.add_argument(
        "--normalize",
        choices=["max", "minmax"],
        default="max",
        help="Normalization per Y series. Default: max (divide by max |y|).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=11,
        help="Smoothing window (Savitzky–Golay if SciPy available; else moving average). Default: 11.",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=2,
        help="Savitzky–Golay polynomial order (ignored for fallback MA). Default: 2.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_dir = args.input_dir.resolve() if args.input_dir else script_dir
    output_path = args.output.resolve() if args.output else (input_dir / "peak_results.xlsx")

    process_folder(
        input_dir=input_dir,
        output_path=output_path,
        normalize=args.normalize,
        window=args.window,
        polyorder=args.polyorder,
    )


if __name__ == "__main__":
    main()
