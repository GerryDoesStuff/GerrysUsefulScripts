#!/usr/bin/env python3
"""
Summarize 2nd columns from all CSVs in THIS SCRIPT'S FOLDER.

- For each input CSV:
  * read header row
  * take the 2nd column's header and append " (<filename>)"
  * take all data from the 2nd column

- Write one output: summary_second_columns.csv
  * Columns correspond to files, in sorted order
  * Rows are positionally aligned; shorter files are padded with empty cells
  * UTF-8 with BOM for Excel compatibility
"""

import os
import csv
import glob

def find_csvs(folder, exclude_name):
    files = []
    for p in glob.glob(os.path.join(folder, "*.csv")):
        if os.path.basename(p).lower() != exclude_name.lower():
            files.append(p)
    return sorted(files, key=lambda s: s.lower())

def read_second_column(path):
    """
    Returns (header_for_second_col, list_of_values_as_strings).
    Skips files with <2 columns or no data rows.
    """
    vals = []
    header2 = None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return None, None  # empty file
        if len(header) < 2:
            return None, None
        header2 = header[1]
        for row in r:
            if len(row) >= 2:
                vals.append(row[1])
            else:
                vals.append("")  # pad missing cell in that row
    return header2, vals

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_name = "summary_second_columns.csv"
    out_path = os.path.join(script_dir, out_name)

    csvs = find_csvs(script_dir, out_name)
    if not csvs:
        print(f"No CSV files found in:\n{script_dir}")
        return

    columns = []   # list of lists (data)
    headers = []   # list of header strings

    for fp in csvs:
        h2, data = read_second_column(fp)
        if h2 is None or data is None:
            print(f"Skipped (no 2nd column): {fp}")
            continue
        fname = os.path.basename(fp)
        headers.append(f"{h2} ({fname})")
        columns.append(data)

    if not columns:
        print("No usable CSVs with a second column.")
        return

    # Align by row index
    max_len = max(len(col) for col in columns)
    # Pad columns to equal length
    for col in columns:
        if len(col) < max_len:
            col.extend([""] * (max_len - len(col)))

    # Transpose columns -> rows
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(max_len):
            w.writerow([columns[j][i] for j in range(len(columns))])

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
