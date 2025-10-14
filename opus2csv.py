#!/usr/bin/env python3
# Extract Bruker OPUS in THIS SCRIPT'S FOLDER to CSV.
# - Keeps numeric suffixes: ".0" -> ".0.csv"
# - If multi-block, appends "_1", "_2", ...
# - Headers: "Wavenumber, cm^-1" and "Intensity, Absorbance"
# - Strips units from data to avoid Pint conversion errors.

import sys, os, glob, csv, io, re

# UTF-8 console + files (Windows-safe)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Readers
_HAVE_BOR = _HAVE_SCP = False
try:
    from brukeropusreader import read_file  # type: ignore
    _HAVE_BOR = True
except Exception:
    pass
try:
    import spectrochempy as scp  # type: ignore
    _HAVE_SCP = True
except Exception:
    pass

if not (_HAVE_BOR or _HAVE_SCP):
    sys.stderr.write("Install one reader:\n  pip install brukeropusreader\n  pip install spectrochempy\n")
    sys.exit(1)

_NUMERIC_EXT_RE = re.compile(r"\.\d+$", re.ASCII)

def candidates_in_dir(dirpath):
    pats = ["*.OPUS","*.opus","*.[0-9]","*.[0-9][0-9]","*.[0-9][0-9][0-9]"]
    out, seen = [], set()
    for pat in pats:
        for p in glob.glob(os.path.join(dirpath, pat)):
            if os.path.isfile(p) and p not in seen:
                seen.add(p); out.append(p)
    return sorted(out)

def out_path_for(src_path, block_index=None):
    # Preserve numeric extension if present
    base = src_path if _NUMERIC_EXT_RE.search(src_path) else os.path.splitext(src_path)[0]
    return f"{base}.csv" if block_index is None else f"{base}_{block_index}.csv"

def normalize_unit(u):
    s = str(u).strip()
    # Common FTIR unit normalizations
    repl = {
        "1 / centimeter": "cm^-1",
        "1/cm": "cm^-1",
        "cm**-1": "cm^-1",
        "1/cm^-1": "cm^-1",
        "None": "",
        "": "",
        "absorbance": "Absorbance",
    }
    return repl.get(s, s)

def to_numeric_array(a):
    # Strip units from SpectroChemPy/Pint objects to plain float array
    import numpy as np
    # Quantity-like
    if hasattr(a, "magnitude"):
        return np.asarray(a.magnitude, dtype=float)
    # SpectroChemPy Coord/Data
    if hasattr(a, "data"):
        return np.asarray(a.data, dtype=float)
    if hasattr(a, "values"):
        v = a.values
        if hasattr(v, "magnitude"):
            return np.asarray(v.magnitude, dtype=float)
        return np.asarray(v, dtype=float)
    # Fallback
    return np.asarray(a, dtype=float)

def write_csv(dst, x_num, y_num, xname, yname):
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([xname, yname])
        for xi, yi in zip(x_num, y_num):
            w.writerow([float(xi), float(yi)])

def read_with_brukeropusreader(path):
    d = read_file(path)
    xunit = normalize_unit(d.get("XUN", "cm^-1"))
    yunit = normalize_unit(d.get("YUN", "Absorbance"))
    fxv, lxv, npt = float(d["FXV"]), float(d["LXV"]), int(d["NPT"])
    y = d["AB"]
    if npt <= 1:
        x = [fxv]
    else:
        step = (lxv - fxv) / (npt - 1)
        x = [fxv + i * step for i in range(npt)]
    if len(y) != npt:
        raise ValueError(f"Length mismatch AB({len(y)}) vs NPT({npt})")
    return [(x, y, xunit, yunit)]

def read_with_spectrochempy(path):
    ds = scp.read_opus(path)
    blocks = []
    def extract_xy(d):
        arr = d.squeeze()
        # x axis and unit
        try:
            x_obj = arr.x
        except Exception:
            x_obj = arr.coordset.x  # type: ignore
        x_vals = to_numeric_array(getattr(x_obj, "values", x_obj))
        x_unit = normalize_unit(getattr(x_obj, "units", ""))
        # y values and unit
        y_vals = to_numeric_array(arr)
        y_unit = normalize_unit(getattr(arr, "units", ""))
        return x_vals, y_vals, x_unit, y_unit
    if hasattr(ds, "__iter__") and not hasattr(ds, "values"):
        for item in ds:
            blocks.append(extract_xy(item))
    else:
        blocks.append(extract_xy(ds))
    return blocks

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = candidates_in_dir(script_dir)
    if not files:
        print(f"No OPUS files found in script folder:\n{script_dir}")
        return
    ok = fail = 0
    for fp in files:
        try:
            blocks = None
            if _HAVE_BOR:
                try:
                    blocks = read_with_brukeropusreader(fp)
                except Exception:
                    blocks = None
            if blocks is None and _HAVE_SCP:
                blocks = read_with_spectrochempy(fp)
            if not blocks:
                raise RuntimeError("No reader could decode this file")

            for idx, (x, y, xunit, yunit) in enumerate(blocks, start=1):
                xname = "Wavenumber" + (", " + xunit if xunit else "")
                yname = "Intensity" + (", " + yunit if yunit else "")
                dst = out_path_for(fp) if len(blocks) == 1 else out_path_for(fp, idx)
                write_csv(dst, to_numeric_array(x), to_numeric_array(y), xname, yname)
                print(f"Wrote {dst}")
            ok += 1
        except Exception as e:
            fail += 1
            sys.stderr.write(f"Failed on {fp}: {e}\n")
    print(f"Done. Success: {ok}. Failed: {fail}.")

if __name__ == "__main__":
    main()
