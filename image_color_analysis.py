#!/usr/bin/env python3
"""
Batch Image Color Analysis → Excel
- Wide-format stats (RGB/HSV/LAB)
- Palette (k-means)
- Histogram tables for RGB, HSV, LAB
- Overlays per image:
    • RGB overlay (R/G/B), x∈[0, plot_xmax], global Y (110%)
    • HSV overlay (H/S/V), x∈[0,1], global Y (110%)
    • LAB overlay (L*/a*/b*), x∈[-128,127], global Y (110%)
- NEW: Normalized aggregate overlays (0–1 A.U.) for RGB/HSV/LAB
    • Plots saved to subfolders and embedded above per-image overlays
    • Optional Agg_Normalized sheet with normalized aggregate values

Install:
    python -m pip install numpy pandas pillow scikit-image scikit-learn xlsxwriter matplotlib
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys, re, math
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --------------------------- Helpers ---------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def natural_key(path: Path):
    s = path.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(files, key=natural_key)

def read_image_rgb_and_mask(path: Path):
    with Image.open(path) as im:
        im = im.convert("RGBA")
        w, h = im.size
        arr = np.asarray(im, dtype=np.uint8)
    rgb = arr[..., :3].astype(np.float32) / 255.0
    alpha = arr[..., 3].astype(np.float32) / 255.0
    mask = alpha > 0
    if not mask.any():
        mask = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)
    return rgb, mask, (w, h)

def flatten_valid(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W, _ = rgb.shape
    flat = rgb.reshape(H * W, 3)
    return flat[mask.reshape(H * W)]

def channel_stats_1d(arr: np.ndarray) -> dict[str, float]:
    arr = arr.astype(np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

def stats_wide(arr: np.ndarray, image_name: str, chan_labels: list[str]) -> pd.DataFrame:
    row = {"image": image_name}
    for i, ch in enumerate(chan_labels):
        s = channel_stats_1d(arr[:, i])
        for k, v in s.items():
            row[f"{k}{ch}"] = v
    return pd.DataFrame([row])

def rgb_to_hex(rgb01: np.ndarray) -> str:
    rgb255 = np.clip(np.round(rgb01 * 255), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb255.tolist())

def compute_hist_counts(arr: np.ndarray, bins: int, rng: tuple[float,float]) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(arr, bins=bins, range=rng)
    return counts.astype(np.int64), edges

# ---------------------- Plotting helpers -----------------------

def plot_rgb_overlay(px_rgb: np.ndarray, bins: int, x_max: float, y_max: int, out_png: Path, title: str):
    fig = plt.figure(figsize=(6.0, 4.0), dpi=120)
    plt.hist(px_rgb[:, 0], bins=bins, range=(0, x_max), histtype="step", label="R", color="red")
    plt.hist(px_rgb[:, 1], bins=bins, range=(0, x_max), histtype="step", label="G", color="green")
    plt.hist(px_rgb[:, 2], bins=bins, range=(0, x_max), histtype="step", label="B", color="blue")
    plt.xlabel(f"Channel value (0–{x_max:g})")
    plt.ylabel("Count")
    plt.title(title)
    plt.ylim(0, y_max)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)

def plot_hsv_overlay(hsv: np.ndarray, bins: int, y_max: int, out_png: Path, title: str):
    fig = plt.figure(figsize=(6.0, 4.0), dpi=120)
    plt.hist(hsv[:, 0], bins=bins, range=(0, 1), histtype="step", label="H", color="black")
    plt.hist(hsv[:, 1], bins=bins, range=(0, 1), histtype="step", label="S", color="orange")
    plt.hist(hsv[:, 2], bins=bins, range=(0, 1), histtype="step", label="V", color="purple")
    plt.xlabel("Value (0–1)")
    plt.ylabel("Count")
    plt.title(title)
    plt.ylim(0, y_max)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)

def plot_lab_overlay(lab: np.ndarray, bins: int, y_max: int, out_png: Path, title: str):
    rng = (-128.0, 127.0)  # common x-range so L*/a*/b* share an axis
    fig = plt.figure(figsize=(6.0, 4.0), dpi=120)
    plt.hist(lab[:, 0], bins=bins, range=rng, histtype="step", label="L*", color="black")
    plt.hist(lab[:, 1], bins=bins, range=rng, histtype="step", label="a*", color="orange")
    plt.hist(lab[:, 2], bins=bins, range=rng, histtype="step", label="b*", color="purple")
    plt.xlabel("Value (common axis [-128…127]; L* lies within 0…100)")
    plt.ylabel("Count")
    plt.title(title)
    plt.ylim(0, y_max)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)

def step_overlay_from_counts(bin_edges: np.ndarray, counts_list: list[np.ndarray], labels: list[str],
                             colors: list[str], y_max: float, out_png: Path, title: str, xlabel: str):
    """Plot overlay using precomputed (normalized) counts via a step curve."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig = plt.figure(figsize=(6.0, 4.0), dpi=120)
    for c, lab, col in zip(counts_list, labels, colors):
        plt.step(centers, c, where="mid", label=lab, color=col)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized count (0–1 A.U.)")
    plt.title(title)
    plt.ylim(0.0, 1.0 if y_max is None else y_max)  # keep 0–1
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)

# ---------------------- Per-image analysis ----------------------

def analyze_image(path: Path, bins: int, clusters: int):
    rgb, mask, (w, h) = read_image_rgb_and_mask(path)
    pixels = flatten_valid(rgb, mask)
    if pixels.size == 0:
        raise ValueError(f"No valid pixels in {path}")

    hsv = color.rgb2hsv(pixels)         # H,S,V in [0,1]
    lab = color.rgb2lab(pixels)         # L*∈[0,100], a*,b*≈[-128,127]

    # Wide stats
    stats_rgb_wide = stats_wide(pixels, path.name, ["R", "G", "B"])
    stats_hsv_wide = stats_wide(hsv,    path.name, ["H", "S", "V"])
    stats_lab_wide = stats_wide(lab,    path.name, ["L", "a", "b"])

    # ΔE2000 vs mean LAB
    mean_lab = lab.mean(axis=0, keepdims=True)
    deltaE = color.deltaE_ciede2000(lab, np.repeat(mean_lab, lab.shape[0], axis=0)).astype(np.float64)
    dE_summary = {
        "image": path.name,
        "width": w, "height": h,
        "pixels_used": int(pixels.shape[0]),
        "deltaE2000_mean": float(np.mean(deltaE)),
        "deltaE2000_p50": float(np.percentile(deltaE, 50)),
        "deltaE2000_p95": float(np.percentile(deltaE, 95)),
        "deltaE2000_max": float(np.max(deltaE)),
    }

    # Palette (k-means on sample)
    N = pixels.shape[0]
    if N > 50000:
        idx = np.random.default_rng(42).choice(N, size=50000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels
    km = KMeans(n_clusters=clusters, n_init=10, random_state=42).fit(sample)
    centers, counts = km.cluster_centers_, np.bincount(km.labels_, minlength=clusters)
    props = counts / counts.sum()
    order = np.argsort(-props)
    palette_rows = []
    for rank, j in enumerate(order, start=1):
        c = centers[j]; hexcode = rgb_to_hex(c)
        R, G, B = np.clip(np.round(c * 255), 0, 255).astype(int).tolist()
        palette_rows.append({"image": path.name, "rank": rank, "hex": hexcode, "R": R, "G": G, "B": B, "proportion": float(props[j])})
    palette_df = pd.DataFrame(palette_rows)[["image","rank","hex","R","G","B","proportion"]]

    # Histogram tables (full ranges for traceability)
    rows_rgb = []
    for i, ch in enumerate(["R","G","B"]):
        counts, _ = compute_hist_counts(pixels[:, i], bins=bins, rng=(0.0, 1.0))
        for b_idx, cnt in enumerate(counts):
            rows_rgb.append({"image": path.name, "channel": ch, "bin": b_idx, "count": int(cnt)})
    hist_rgb_df = pd.DataFrame(rows_rgb)

    rows_hsv = []
    for i, ch in enumerate(["H","S","V"]):
        counts, _ = compute_hist_counts(hsv[:, i], bins=bins, rng=(0.0, 1.0))
        for b_idx, cnt in enumerate(counts):
            rows_hsv.append({"image": path.name, "channel": ch, "bin": b_idx, "count": int(cnt)})
    hist_hsv_df = pd.DataFrame(rows_hsv)

    rows_lab = []
    lab_ranges = [(0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0)]
    for i, (ch, rng) in enumerate(zip(["L","a","b"], lab_ranges)):
        counts, _ = compute_hist_counts(lab[:, i], bins=bins, rng=rng)
        for b_idx, cnt in enumerate(counts):
            rows_lab.append({"image": path.name, "channel": ch, "bin": b_idx, "count": int(cnt)})
    hist_lab_df = pd.DataFrame(rows_lab)

    return {
        "pixels": pixels, "hsv": hsv, "lab": lab,
        "summary": dE_summary,
        "stats_rgb_wide": stats_rgb_wide,
        "stats_hsv_wide": stats_hsv_wide,
        "stats_lab_wide": stats_lab_wide,
        "palette": palette_df,
        "hist_rgb": hist_rgb_df,
        "hist_hsv": hist_hsv_df,
        "hist_lab": hist_lab_df,
    }

# --------------- Global Y-limits for each histogram family ---------------

def global_y_rgb(images: list[Path], bins: int, x_max: float) -> int:
    g = 0
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            for i in (0,1,2):
                counts, _ = np.histogram(px[:, i], bins=bins, range=(0.0, x_max))
                if counts.size: g = max(g, int(counts.max()))
        except Exception: pass
    return max(1, int(math.ceil(g * 1.10)))

def global_y_hsv(images: list[Path], bins: int) -> int:
    g = 0
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            hsv = color.rgb2hsv(px)
            for i in (0,1,2):
                counts, _ = np.histogram(hsv[:, i], bins=bins, range=(0.0, 1.0))
                if counts.size: g = max(g, int(counts.max()))
        except Exception: pass
    return max(1, int(math.ceil(g * 1.10)))

def global_y_lab(images: list[Path], bins: int) -> int:
    g = 0
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            lab = color.rgb2lab(px)
            rng = (-128.0, 127.0)
            for i in (0,1,2):
                counts, _ = np.histogram(lab[:, i], bins=bins, range=rng)
                if counts.size: g = max(g, int(counts.max()))
        except Exception: pass
    return max(1, int(math.ceil(g * 1.10)))

# ---------------- Aggregated (normalized) histograms ----------------

def aggregate_counts_rgb(images: list[Path], bins: int, x_max: float):
    agg = [np.zeros(bins, dtype=np.int64) for _ in range(3)]
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            for i in (0,1,2):
                c, edges = np.histogram(px[:, i], bins=bins, range=(0.0, x_max))
                agg[i] += c.astype(np.int64)
        except Exception: pass
    return edges, agg  # edges from last iter is fine since identical bins

def aggregate_counts_hsv(images: list[Path], bins: int):
    agg = [np.zeros(bins, dtype=np.int64) for _ in range(3)]
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            hsv = color.rgb2hsv(px)
            for i in (0,1,2):
                c, edges = np.histogram(hsv[:, i], bins=bins, range=(0.0, 1.0))
                agg[i] += c.astype(np.int64)
        except Exception: pass
    return edges, agg

def aggregate_counts_lab(images: list[Path], bins: int):
    rng = (-128.0, 127.0)
    agg = [np.zeros(bins, dtype=np.int64) for _ in range(3)]
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            lab = color.rgb2lab(px)
            for i in (0,1,2):
                c, edges = np.histogram(lab[:, i], bins=bins, range=rng)
                agg[i] += c.astype(np.int64)
        except Exception: pass
    return edges, agg

def normalize_0_1_across_channels(agg_list: list[np.ndarray]) -> list[np.ndarray]:
    maxval = max(int(a.max()) if a.size else 0 for a in agg_list)
    if maxval <= 0:
        return [a.astype(np.float64) for a in agg_list]
    return [a.astype(np.float64) / float(maxval) for a in agg_list]

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch image color analysis → Excel (stats + palettes + histograms + overlays + aggregates)")
    parser.add_argument("--bins", type=int, default=256, help="Histogram bins (default: 256)")
    parser.add_argument("--clusters", type=int, default=5, help="K-means palette size (default: 5)")
    parser.add_argument("--output", type=str, default="image_color_analysis.xlsx", help="Output Excel filename")
    parser.add_argument("--plots-dir", type=str, default="__plots__", help="Base directory to save histogram PNGs")
    parser.add_argument("--subdir-rgb", type=str, default="rgb", help="Subfolder under plots-dir for RGB plots")
    parser.add_argument("--subdir-hue", type=str, default="hue", help="Subfolder under plots-dir for HSV plots")
    parser.add_argument("--subdir-lab", type=str, default="lab", help="Subfolder under plots-dir for LAB plots")
    parser.add_argument("--plot-xmax", type=float, default=0.97, help="Upper x-limit for RGB plotted histograms")
    parser.add_argument("--write-agg-table", action="store_true",
                        help="Also write normalized aggregate values to 'Agg_Normalized' sheet")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    images = list_images(script_dir)
    if not images:
        print("No images found next to this script.", file=sys.stderr); sys.exit(1)

    print(f"Found {len(images)} image(s). Analyzing in natural order...")

    # Pass 1: analysis
    summary_rows, stats_rgb_rows, stats_hsv_rows, stats_lab_rows = [], [], [], []
    palette_all, hist_rgb_all, hist_hsv_all, hist_lab_all = [], [], [], []
    per_image_data = []

    for i, path in enumerate(images, start=1):
        try:
            print(f"[{i}/{len(images)}] {path.name}")
            res = analyze_image(path, bins=args.bins, clusters=args.clusters)
            per_image_data.append((path, res["pixels"], res["hsv"], res["lab"]))
            summary_rows.append(res["summary"])
            stats_rgb_rows.append(res["stats_rgb_wide"])
            stats_hsv_rows.append(res["stats_hsv_wide"])
            stats_lab_rows.append(res["stats_lab_wide"])
            palette_all.append(res["palette"])
            hist_rgb_all.append(res["hist_rgb"])
            hist_hsv_all.append(res["hist_hsv"])
            hist_lab_all.append(res["hist_lab"])
        except Exception as e:
            print(f"  ! Skipping {path.name} due to error: {e}", file=sys.stderr)

    # Concatenate tables
    summary_df   = pd.DataFrame(summary_rows)
    stats_rgb_df = pd.concat(stats_rgb_rows, ignore_index=True) if stats_rgb_rows else pd.DataFrame()
    stats_hsv_df = pd.concat(stats_hsv_rows, ignore_index=True) if stats_hsv_rows else pd.DataFrame()
    stats_lab_df = pd.concat(stats_lab_rows, ignore_index=True) if stats_lab_rows else pd.DataFrame()
    palette_df   = pd.concat(palette_all,    ignore_index=True) if palette_all    else pd.DataFrame()
    rgb_hist_df  = pd.concat(hist_rgb_all,   ignore_index=True) if hist_rgb_all   else pd.DataFrame()
    hsv_hist_df  = pd.concat(hist_hsv_all,   ignore_index=True) if hist_hsv_all   else pd.DataFrame()
    lab_hist_df  = pd.concat(hist_lab_all,   ignore_index=True) if hist_lab_all   else pd.DataFrame()

    # Global Y limits for per-image overlays
    print("Computing global Y limits for per-image overlays...")
    y_max_rgb = global_y_rgb(images, bins=args.bins, x_max=args.plot_xmax)
    y_max_hsv = global_y_hsv(images, bins=args.bins)
    y_max_lab = global_y_lab(images, bins=args.bins)
    print(f"Ymax RGB={y_max_rgb}, HSV={y_max_hsv}, LAB={y_max_lab}")

    # Pass 2: generate per-image overlay plots into subfolders
    plots_base = script_dir / args.plots_dir
    dir_rgb = plots_base / args.subdir_rgb
    dir_hsv = plots_base / args.subdir_hue   # hue=HSV
    dir_lab = plots_base / args.subdir_lab
    dir_rgb.mkdir(parents=True, exist_ok=True)
    dir_hsv.mkdir(parents=True, exist_ok=True)
    dir_lab.mkdir(parents=True, exist_ok=True)

    plot_paths_rgb, plot_paths_hsv, plot_paths_lab = [], [], []

    for path, px, hsv, lab in per_image_data:
        p_rgb = dir_rgb / f"{path.stem}.png"
        plot_rgb_overlay(px,  bins=args.bins, x_max=args.plot_xmax, y_max=y_max_rgb, out_png=p_rgb, title=path.name)
        plot_paths_rgb.append((path.name, p_rgb))

        p_hsv = dir_hsv / f"{path.stem}.png"
        plot_hsv_overlay(hsv, bins=args.bins, y_max=y_max_hsv, out_png=p_hsv, title=path.name)
        plot_paths_hsv.append((path.name, p_hsv))

        p_lab = dir_lab / f"{path.stem}.png"
        plot_lab_overlay(lab, bins=args.bins, y_max=y_max_lab, out_png=p_lab, title=path.name)
        plot_paths_lab.append((path.name, p_lab))

    # ----------------- Aggregated, normalized overlays -----------------
    print("Computing normalized aggregate histograms (0–1 A.U.)...")
    # RGB (respect x_max)
    edges_rgb, agg_rgb_raw = aggregate_counts_rgb(images, bins=args.bins, x_max=args.plot_xmax)
    agg_rgb_norm = normalize_0_1_across_channels(agg_rgb_raw)
    p_rgb_agg = dir_rgb / "_aggregate.png"
    step_overlay_from_counts(
        edges_rgb, agg_rgb_norm, labels=["R", "G", "B"], colors=["red", "green", "blue"],
        y_max=1.0, out_png=p_rgb_agg,
        title=f"Aggregate RGB (x∈[0,{args.plot_xmax:g}], normalized 0–1 A.U.)",
        xlabel=f"Channel value (0–{args.plot_xmax:g})"
    )

    # HSV
    edges_hsv, agg_hsv_raw = aggregate_counts_hsv(images, bins=args.bins)
    agg_hsv_norm = normalize_0_1_across_channels(agg_hsv_raw)
    p_hsv_agg = dir_hsv / "_aggregate.png"
    step_overlay_from_counts(
        edges_hsv, agg_hsv_norm, labels=["H", "S", "V"], colors=["black", "orange", "purple"],
        y_max=1.0, out_png=p_hsv_agg,
        title="Aggregate HSV (normalized 0–1 A.U.)",
        xlabel="Value (0–1)"
    )

    # LAB (common [-128,127])
    edges_lab, agg_lab_raw = aggregate_counts_lab(images, bins=args.bins)
    agg_lab_norm = normalize_0_1_across_channels(agg_lab_raw)
    p_lab_agg = dir_lab / "_aggregate.png"
    step_overlay_from_counts(
        edges_lab, agg_lab_norm, labels=["L*", "a*", "b*"], colors=["black", "orange", "purple"],
        y_max=1.0, out_png=p_lab_agg,
        title="Aggregate LAB (normalized 0–1 A.U.)",
        xlabel="Value (common axis [-128…127])"
    )

    # Optional aggregate table
    agg_table_df = None
    if args.write_agg_table:
        def mk_df(space: str, labels: list[str], edges: np.ndarray, norm_counts: list[np.ndarray]) -> pd.DataFrame:
            centers = 0.5 * (edges[:-1] + edges[1:])
            rows = []
            for ch, vals in zip(labels, norm_counts):
                for b, v in enumerate(vals):
                    rows.append({"space": space, "channel": ch, "bin": b, "center": float(centers[b]), "value": float(v)})
            return pd.DataFrame(rows)
        df_rgb = mk_df("RGB", ["R","G","B"], edges_rgb, agg_rgb_norm)
        df_hsv = mk_df("HSV", ["H","S","V"], edges_hsv, agg_hsv_norm)
        df_lab = mk_df("LAB", ["L*","a*","b*"], edges_lab, agg_lab_norm)
        agg_table_df = pd.concat([df_rgb, df_hsv, df_lab], ignore_index=True)

    # ---------------- Write Excel + embed images ----------------
    out_path = script_dir / args.output
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Tables
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        stats_rgb_df.to_excel(writer, index=False, sheet_name="ChannelStats_RGB")
        stats_hsv_df.to_excel(writer, index=False, sheet_name="ChannelStats_HSV")
        stats_lab_df.to_excel(writer, index=False, sheet_name="ChannelStats_LAB")
        palette_df.to_excel(writer, index=False, sheet_name="Palette")
        rgb_hist_df.to_excel(writer, index=False, sheet_name="Hist_RGB")
        hsv_hist_df.to_excel(writer, index=False, sheet_name="Hist_HSV")
        lab_hist_df.to_excel(writer, index=False, sheet_name="Hist_LAB")
        if agg_table_df is not None:
            agg_table_df.to_excel(writer, index=False, sheet_name="Agg_Normalized")

        # Column widths
        for name, sheet in writer.sheets.items():
            try:
                if name in {"ChannelStats_RGB","ChannelStats_HSV","ChannelStats_LAB","Palette","Summary","Agg_Normalized"}:
                    sheet.set_column(0, 0, 28); sheet.set_column(1, 50, 14)
                else:
                    sheet.set_column(0, 0, 28); sheet.set_column(1, 4, 12)
            except Exception: pass

        # Place overlays under each table: first the AGGREGATE, then per-image
        stride = 28; col = 0

        # RGB
        ws = writer.sheets["Hist_RGB"]
        if len(rgb_hist_df) > 0:
            base = len(rgb_hist_df) + 4
            # Aggregate first
            ws.write(base - 1, col, f"AGGREGATE RGB (normalized 0–1 A.U., 0–{args.plot_xmax:g})")
            ws.insert_image(base, col, str(p_rgb_agg))
            row = base + stride
            # Then per-image
            for name, png in plot_paths_rgb:
                ws.write(row - 1, col, f"RGB overlay: {name}")
                ws.insert_image(row, col, str(png))
                row += stride

        # HSV
        ws = writer.sheets["Hist_HSV"]
        if len(hsv_hist_df) > 0:
            base = len(hsv_hist_df) + 4
            ws.write(base - 1, col, "AGGREGATE HSV (normalized 0–1 A.U.)")
            ws.insert_image(base, col, str(p_hsv_agg))
            row = base + stride
            for name, png in plot_paths_hsv:
                ws.write(row - 1, col, f"HSV overlay: {name}")
                ws.insert_image(row, col, str(png))
                row += stride

        # LAB
        ws = writer.sheets["Hist_LAB"]
        if len(lab_hist_df) > 0:
            base = len(lab_hist_df) + 4
            ws.write(base - 1, col, "AGGREGATE LAB (normalized 0–1 A.U.)")
            ws.insert_image(base, col, str(p_lab_agg))
            row = base + stride
            for name, png in plot_paths_lab:
                ws.write(row - 1, col, f"LAB overlay: {name}")
                ws.insert_image(row, col, str(png))
                row += stride

    print(f"Done. Wrote {out_path.name}")

if __name__ == "__main__":
    main()
