#!/usr/bin/env python3
"""
Batch Image Color Analysis → Excel

Adds:
- 2D intensity maps for the “all-images overlays” (per channel in RGB/HSV/LAB):
  X = bin index, Y = image index, color = histogram counts.
  Saved as *_map.png and placed alongside the overlay plots in Excel.

Existing features:
- Natural filename order
- Wide-format stats (RGB/HSV/LAB)
- ΔE2000 vs mean LAB (Summary)
- K-means palette
- Histogram tables (RGB, HSV, LAB)
- Per-image overlays (RGB clipped to 0–plot_xmax; HSV/LAB full ranges) with global Y=110% max
- Normalized aggregate overlays (0–1 A.U.) for RGB/HSV/LAB
- Channel-wise “all-images overlays” (rainbow from red→violet across images)
- Plots saved to subfolders; embedded under tables
- Optional aggregate table (--write-agg-table)

Deps:
    python -m pip install numpy pandas pillow scikit-image scikit-learn xlsxwriter matplotlib
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys, re, math, colorsys
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
    plt.xlabel(f"Channel value (0–{x_max:g})"); plt.ylabel("Count"); plt.title(title)
    plt.ylim(0, y_max); plt.legend(); plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()

def plot_hsv_overlay(hsv: np.ndarray, bins: int, y_max: int, out_png: Path, title: str):
    fig = plt.figure(figsize=(6.0, 4.0), dpi=120)
    plt.hist(hsv[:, 0], bins=bins, range=(0, 1), histtype="step", label="H", color="black")
    plt.hist(hsv[:, 1], bins=bins, range=(0, 1), histtype="step", label="S", color="orange")
    plt.hist(hsv[:, 2], bins=bins, range=(0, 1), histtype="step", label="V", color="purple")
    plt.xlabel("Value (0–1)"); plt.ylabel("Count"); plt.title(title)
    plt.ylim(0, y_max); plt.legend(); plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()

def plot_lab_overlay(lab: np.ndarray, bins: int, y_max: int, out_png: Path, title: str):
    rng = (-128.0, 127.0)
    fig = plt.figure(figsize=(6.0, 4.0), dpi=120)
    plt.hist(lab[:, 0], bins=bins, range=rng, histtype="step", label="L*", color="black")
    plt.hist(lab[:, 1], bins=bins, range=rng, histtype="step", label="a*", color="orange")
    plt.hist(lab[:, 2], bins=bins, range=rng, histtype="step", label="b*", color="purple")
    plt.xlabel("Value (common axis [-128…127]; L* within 0…100)"); plt.ylabel("Count"); plt.title(title)
    plt.ylim(0, y_max); plt.legend(); plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()

def step_overlay_from_counts(edges: np.ndarray, counts_list: list[np.ndarray], labels: list[str],
                             colors: list[tuple[float,float,float]], y_max: float, out_png: Path,
                             title: str, xlabel: str, show_legend: bool = True):
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = plt.figure(figsize=(6.0, 4.0), dpi=120)
    for c, lab, col in zip(counts_list, labels, colors):
        plt.step(centers, c, where="mid", label=lab, color=col)
    plt.xlabel(xlabel); plt.ylabel("Normalized count (0–1 A.U.)" if y_max <= 1.01 else "Count"); plt.title(title)
    plt.ylim(0.0, y_max)
    if show_legend: plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()

def rainbow_colors(n: int) -> list[tuple[float, float, float]]:
    if n <= 1: return [(1.0, 0.0, 0.0)]
    return [colorsys.hsv_to_rgb(0.0 + 0.8 * i / (n - 1), 1.0, 1.0) for i in range(n)]

# -------- New: counts matrix + heatmap (X=bin index, Y=image index, color=counts) --------

def channel_counts_matrix(per_image_data, extractor, bins: int, rng: tuple[float,float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (edges, M) where M has shape (num_images, bins) with raw counts per image.
    """
    num_images = len(per_image_data)
    M = np.zeros((num_images, bins), dtype=np.int64)
    edges = None
    for row, (_, px, hsv, lab) in enumerate(per_image_data):
        vals = extractor(px, hsv, lab)
        counts, edges = np.histogram(vals, bins=bins, range=rng)
        M[row, :] = counts.astype(np.int64)
    return edges, M

def plot_counts_heatmap(M: np.ndarray, out_png: Path, title: str, xlabel: str = "Bin index", ylabel: str = "Image index"):
    """
    Simple intensity map using imshow; color encodes raw counts. No custom colormap specified.
    """
    fig = plt.figure(figsize=(6.0, 4.0), dpi=120)
    ax = plt.gca()
    im = ax.imshow(M, aspect="auto", origin="lower")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()

# ---------------------- Per-image analysis ----------------------

def analyze_image(path: Path, bins: int, clusters: int):
    rgb, mask, (w, h) = read_image_rgb_and_mask(path)
    pixels = flatten_valid(rgb, mask)
    if pixels.size == 0:
        raise ValueError(f"No valid pixels in {path}")

    hsv = color.rgb2hsv(pixels)
    lab = color.rgb2lab(pixels)

    stats_rgb_wide = stats_wide(pixels, path.name, ["R", "G", "B"])
    stats_hsv_wide = stats_wide(hsv,    path.name, ["H", "S", "V"])
    stats_lab_wide = stats_wide(lab,    path.name, ["L", "a", "b"])

    mean_lab = lab.mean(axis=0, keepdims=True)
    deltaE = color.deltaE_ciede2000(lab, np.repeat(mean_lab, lab.shape[0], axis=0)).astype(np.float64)
    dE_summary = {
        "image": path.name, "width": w, "height": h, "pixels_used": int(pixels.shape[0]),
        "deltaE2000_mean": float(np.mean(deltaE)), "deltaE2000_p50": float(np.percentile(deltaE, 50)),
        "deltaE2000_p95": float(np.percentile(deltaE, 95)), "deltaE2000_max": float(np.max(deltaE)),
    }

    # Palette via KMeans on sample
    N = pixels.shape[0]
    sample = pixels[np.random.default_rng(42).choice(N, size=50000, replace=False)] if N > 50000 else pixels
    km = KMeans(n_clusters=clusters, n_init=10, random_state=42).fit(sample)
    centers, counts = km.cluster_centers_, np.bincount(km.labels_, minlength=clusters)
    props = counts / counts.sum(); order = np.argsort(-props)
    palette_rows = []
    for rank, j in enumerate(order, start=1):
        c = centers[j]; hexcode = rgb_to_hex(c)
        R, G, B = np.clip(np.round(c * 255), 0, 255).astype(int).tolist()
        palette_rows.append({"image": path.name, "rank": rank, "hex": hexcode, "R": R, "G": G, "B": B, "proportion": float(props[j])})
    palette_df = pd.DataFrame(palette_rows)[["image","rank","hex","R","G","B","proportion"]]

    # Hist tables (full ranges)
    rows_rgb, rows_hsv, rows_lab = [], [], []
    for i, ch in enumerate(["R","G","B"]):
        c, _ = compute_hist_counts(pixels[:, i], bins=bins, rng=(0.0, 1.0))
        rows_rgb += [{"image": path.name, "channel": ch, "bin": b, "count": int(cnt)} for b, cnt in enumerate(c)]
    for i, ch in enumerate(["H","S","V"]):
        c, _ = compute_hist_counts(hsv[:, i], bins=bins, rng=(0.0, 1.0))
        rows_hsv += [{"image": path.name, "channel": ch, "bin": b, "count": int(cnt)} for b, cnt in enumerate(c)]
    for (ch, rng) in zip(["L","a","b"], [(0.0,100.0), (-128.0,127.0), (-128.0,127.0)]):
        c, _ = compute_hist_counts(lab[:, {"L":0,"a":1,"b":2}[ch]], bins=bins, rng=rng)
        rows_lab += [{"image": path.name, "channel": ch, "bin": b, "count": int(cnt)} for b, cnt in enumerate(c)]

    return {
        "pixels": pixels, "hsv": hsv, "lab": lab,
        "summary": dE_summary,
        "stats_rgb_wide": stats_rgb_wide, "stats_hsv_wide": stats_hsv_wide, "stats_lab_wide": stats_lab_wide,
        "palette": palette_df,
        "hist_rgb": pd.DataFrame(rows_rgb),
        "hist_hsv": pd.DataFrame(rows_hsv),
        "hist_lab": pd.DataFrame(rows_lab),
    }

# ---------------- Global Y-limits for overlays ----------------

def global_y_rgb(images: list[Path], bins: int, x_max: float) -> int:
    g = 0
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            for i in (0,1,2):
                counts, _ = np.histogram(px[:, i], bins=bins, range=(0.0, x_max))
                g = max(g, int(counts.max()))
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
                g = max(g, int(counts.max()))
        except Exception: pass
    return max(1, int(math.ceil(g * 1.10)))

def global_y_lab(images: list[Path], bins: int) -> int:
    g = 0
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            lab = color.rgb2lab(px); rng = (-128.0, 127.0)
            for i in (0,1,2):
                counts, _ = np.histogram(lab[:, i], bins=bins, range=rng)
                g = max(g, int(counts.max()))
        except Exception: pass
    return max(1, int(math.ceil(g * 1.10)))

# ---------------- Aggregated (normalized) histograms ----------------

def aggregate_counts_rgb(images: list[Path], bins: int, x_max: float):
    agg = [np.zeros(bins, dtype=np.int64) for _ in range(3)]; edges = None
    for p in images:
        try:
            rgb, mask, _ = read_image_rgb_and_mask(p); px = flatten_valid(rgb, mask)
            for i in (0,1,2):
                c, edges = np.histogram(px[:, i], bins=bins, range=(0.0, x_max))
                agg[i] += c.astype(np.int64)
        except Exception: pass
    return edges, agg

def aggregate_counts_hsv(images: list[Path], bins: int):
    agg = [np.zeros(bins, dtype=np.int64) for _ in range(3)]; edges = None
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
    agg = [np.zeros(bins, dtype=np.int64) for _ in range(3)]; edges = None
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
    return [a.astype(np.float64) / float(maxval) if maxval > 0 else a.astype(np.float64) for a in agg_list]

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch image color analysis → Excel (stats + palettes + histograms + overlays + aggregates + channel overlays + heatmaps)")
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
            stats_rgb_rows.append(res["stats_rgb_wide"]); stats_hsv_rows.append(res["stats_hsv_wide"]); stats_lab_rows.append(res["stats_lab_wide"])
            palette_all.append(res["palette"])
            hist_rgb_all.append(res["hist_rgb"]); hist_hsv_all.append(res["hist_hsv"]); hist_lab_all.append(res["hist_lab"])
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

    # Global Y limits for overlays
    print("Computing global Y limits for per-image overlays...")
    y_max_rgb = global_y_rgb(images, bins=args.bins, x_max=args.plot_xmax)
    y_max_hsv = global_y_hsv(images, bins=args.bins)
    y_max_lab = global_y_lab(images, bins=args.bins)
    print(f"Ymax RGB={y_max_rgb}, HSV={y_max_hsv}, LAB={y_max_lab}")

    # Plot dirs
    plots_base = script_dir / args.plots_dir
    dir_rgb = plots_base / args.subdir_rgb
    dir_hsv = plots_base / args.subdir_hue
    dir_lab = plots_base / args.subdir_lab
    dir_rgb.mkdir(parents=True, exist_ok=True); dir_hsv.mkdir(parents=True, exist_ok=True); dir_lab.mkdir(parents=True, exist_ok=True)

    # Per-image overlays
    plot_paths_rgb, plot_paths_hsv, plot_paths_lab = [], [], []
    for path, px, hsv, lab in per_image_data:
        p_rgb = dir_rgb / f"{path.stem}.png"; plot_rgb_overlay(px, args.bins, args.plot_xmax, y_max_rgb, p_rgb, path.name); plot_paths_rgb.append((path.name, p_rgb))
        p_hsv = dir_hsv / f"{path.stem}.png"; plot_hsv_overlay(hsv, args.bins, y_max_hsv, p_hsv, path.name); plot_paths_hsv.append((path.name, p_hsv))
        p_lab = dir_lab / f"{path.stem}.png"; plot_lab_overlay(lab, args.bins, y_max_lab, p_lab, path.name); plot_paths_lab.append((path.name, p_lab))

    # Aggregates (normalized 0–1 A.U.)
    print("Computing normalized aggregates...")
    edges_rgb, agg_rgb_raw = aggregate_counts_rgb(images, args.bins, args.plot_xmax)
    edges_hsv, agg_hsv_raw = aggregate_counts_hsv(images, args.bins)
    edges_lab, agg_lab_raw = aggregate_counts_lab(images, args.bins)
    p_rgb_agg = dir_rgb / "_aggregate.png"; step_overlay_from_counts(edges_rgb, normalize_0_1_across_channels(agg_rgb_raw), ["R","G","B"], [(1,0,0),(0,1,0),(0,0,1)], 1.0, p_rgb_agg, f"Aggregate RGB (0–{args.plot_xmax:g})", f"Channel value (0–{args.plot_xmax:g})")
    p_hsv_agg = dir_hsv / "_aggregate.png"; step_overlay_from_counts(edges_hsv, normalize_0_1_across_channels(agg_hsv_raw), ["H","S","V"], [(0,0,0),(1,0.5,0),(0.5,0,0.5)], 1.0, p_hsv_agg, "Aggregate HSV", "Value (0–1)")
    p_lab_agg = dir_lab / "_aggregate.png"; step_overlay_from_counts(edges_lab, normalize_0_1_across_channels(agg_lab_raw), ["L*","a*","b*"], [(0,0,0),(1,0.5,0),(0.5,0,0.5)], 1.0, p_lab_agg, "Aggregate LAB", "Value ([-128…127])")

    # Channel-wise all-images overlays (rainbow)
    print("Building all-images overlays + heatmaps…")
    rainbow = rainbow_colors(len(per_image_data))

    # RGB channels
    def ch_counts(extractor, rng): return channel_counts_matrix(per_image_data, extractor, args.bins, rng)
    # R
    edges, counts_list = [], []
    edges_R, M_R = ch_counts(lambda px,hsv,lab: px[:,0], (0.0, args.plot_xmax))
    counts_list = [M_R[i,:] for i in range(M_R.shape[0])]
    p_rgb_R_all = dir_rgb / "_allimages_R.png"
    step_overlay_from_counts(edges_R, counts_list, [f"{i+1}" for i in range(M_R.shape[0])], rainbow, y_max_rgb, p_rgb_R_all, f"All-images overlay — R (0–{args.plot_xmax:g})", f"R value (0–{args.plot_xmax:g})", show_legend=False)
    p_rgb_R_map = dir_rgb / "_allimages_R_map.png"; plot_counts_heatmap(M_R, p_rgb_R_map, "All-images heatmap — R", "Bin index", "Image index")
    # G
    edges_G, M_G = ch_counts(lambda px,hsv,lab: px[:,1], (0.0, args.plot_xmax))
    p_rgb_G_all = dir_rgb / "_allimages_G.png"
    step_overlay_from_counts(edges_G, [M_G[i,:] for i in range(M_G.shape[0])], [f"{i+1}" for i in range(M_G.shape[0])], rainbow, y_max_rgb, p_rgb_G_all, f"All-images overlay — G (0–{args.plot_xmax:g})", f"G value (0–{args.plot_xmax:g})", show_legend=False)
    p_rgb_G_map = dir_rgb / "_allimages_G_map.png"; plot_counts_heatmap(M_G, p_rgb_G_map, "All-images heatmap — G", "Bin index", "Image index")
    # B
    edges_B, M_B = ch_counts(lambda px,hsv,lab: px[:,2], (0.0, args.plot_xmax))
    p_rgb_B_all = dir_rgb / "_allimages_B.png"
    step_overlay_from_counts(edges_B, [M_B[i,:] for i in range(M_B.shape[0])], [f"{i+1}" for i in range(M_B.shape[0])], rainbow, y_max_rgb, p_rgb_B_all, f"All-images overlay — B (0–{args.plot_xmax:g})", f"B value (0–{args.plot_xmax:g})", show_legend=False)
    p_rgb_B_map = dir_rgb / "_allimages_B_map.png"; plot_counts_heatmap(M_B, p_rgb_B_map, "All-images heatmap — B", "Bin index", "Image index")

    # HSV channels (0–1)
    edges_H, M_H = ch_counts(lambda px,hsv,lab: hsv[:,0], (0.0, 1.0))
    p_hsv_H_all = dir_hsv / "_allimages_H.png"
    step_overlay_from_counts(edges_H, [M_H[i,:] for i in range(M_H.shape[0])], [f"{i+1}" for i in range(M_H.shape[0])], rainbow, y_max_hsv, p_hsv_H_all, "All-images overlay — H (0–1)", "Hue (0–1)", show_legend=False)
    p_hsv_H_map = dir_hsv / "_allimages_H_map.png"; plot_counts_heatmap(M_H, p_hsv_H_map, "All-images heatmap — H", "Bin index", "Image index")

    edges_S, M_S = ch_counts(lambda px,hsv,lab: hsv[:,1], (0.0, 1.0))
    p_hsv_S_all = dir_hsv / "_allimages_S.png"
    step_overlay_from_counts(edges_S, [M_S[i,:] for i in range(M_S.shape[0])], [f"{i+1}" for i in range(M_S.shape[0])], rainbow, y_max_hsv, p_hsv_S_all, "All-images overlay — S (0–1)", "Saturation (0–1)", show_legend=False)
    p_hsv_S_map = dir_hsv / "_allimages_S_map.png"; plot_counts_heatmap(M_S, p_hsv_S_map, "All-images heatmap — S", "Bin index", "Image index")

    edges_V, M_V = ch_counts(lambda px,hsv,lab: hsv[:,2], (0.0, 1.0))
    p_hsv_V_all = dir_hsv / "_allimages_V.png"
    step_overlay_from_counts(edges_V, [M_V[i,:] for i in range(M_V.shape[0])], [f"{i+1}" for i in range(M_V.shape[0])], rainbow, y_max_hsv, p_hsv_V_all, "All-images overlay — V (0–1)", "Value (0–1)", show_legend=False)
    p_hsv_V_map = dir_hsv / "_allimages_V_map.png"; plot_counts_heatmap(M_V, p_hsv_V_map, "All-images heatmap — V", "Bin index", "Image index")

    # LAB channels
    edges_L, M_L = ch_counts(lambda px,hsv,lab: lab[:,0], (0.0, 100.0))
    p_lab_L_all = dir_lab / "_allimages_L.png"
    step_overlay_from_counts(edges_L, [M_L[i,:] for i in range(M_L.shape[0])], [f"{i+1}" for i in range(M_L.shape[0])], rainbow, y_max_lab, p_lab_L_all, "All-images overlay — L* (0–100)", "L* (0–100)", show_legend=False)
    p_lab_L_map = dir_lab / "_allimages_L_map.png"; plot_counts_heatmap(M_L, p_lab_L_map, "All-images heatmap — L*", "Bin index", "Image index")

    edges_a, M_a = ch_counts(lambda px,hsv,lab: lab[:,1], (-128.0, 127.0))
    p_lab_a_all = dir_lab / "_allimages_a.png"
    step_overlay_from_counts(edges_a, [M_a[i,:] for i in range(M_a.shape[0])], [f"{i+1}" for i in range(M_a.shape[0])], rainbow, y_max_lab, p_lab_a_all, "All-images overlay — a* (-128…127)", "a* (-128…127)", show_legend=False)
    p_lab_a_map = dir_lab / "_allimages_a_map.png"; plot_counts_heatmap(M_a, p_lab_a_map, "All-images heatmap — a*", "Bin index", "Image index")

    edges_b, M_b = ch_counts(lambda px,hsv,lab: lab[:,2], (-128.0, 127.0))
    p_lab_b_all = dir_lab / "_allimages_b.png"
    step_overlay_from_counts(edges_b, [M_b[i,:] for i in range(M_b.shape[0])], [f"{i+1}" for i in range(M_b.shape[0])], rainbow, y_max_lab, p_lab_b_all, "All-images overlay — b* (-128…127)", "b* (-128…127)", show_legend=False)
    p_lab_b_map = dir_lab / "_allimages_b_map.png"; plot_counts_heatmap(M_b, p_lab_b_map, "All-images heatmap — b*", "Bin index", "Image index")

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

        # Optional aggregate normalized table
        if args.write_agg_table:
            def mk_df(space: str, labels: list[str], edges, counts_norm):
                centers = 0.5 * (edges[:-1] + edges[1:])
                rows = []
                for labc, vals in zip(labels, counts_norm):
                    for b, v in enumerate(vals):
                        rows.append({"space": space, "channel": labc, "bin": b, "center": float(centers[b]), "value": float(v)})
                return pd.DataFrame(rows)
            agg_table_df = pd.concat([
                mk_df("RGB", ["R","G","B"], edges_rgb, normalize_0_1_across_channels(agg_rgb_raw)),
                mk_df("HSV", ["H","S","V"], edges_hsv, normalize_0_1_across_channels(agg_hsv_raw)),
                mk_df("LAB", ["L*","a*","b*"], edges_lab, normalize_0_1_across_channels(agg_lab_raw)),
            ], ignore_index=True)
            agg_table_df.to_excel(writer, index=False, sheet_name="Agg_Normalized")

        # Column widths for data sheets
        for name, sheet in writer.sheets.items():
            try:
                if name in {"ChannelStats_RGB","ChannelStats_HSV","ChannelStats_LAB","Palette","Summary","Agg_Normalized"}:
                    sheet.set_column(0, 0, 28); sheet.set_column(1, 50, 14)
                else:
                    sheet.set_column(0, 0, 28); sheet.set_column(1, 4, 12)
            except Exception: pass

        # Place overlays under each table: Aggregate → Channel overlays (+ heatmaps) → Per-image
        stride = 28; col_left = 0; col_right = 9  # put heatmaps in a right-hand column

        # RGB
        ws = writer.sheets["Hist_RGB"]
        if len(rgb_hist_df) > 0:
            base = len(rgb_hist_df) + 4
            ws.write(base - 1, col_left, f"AGGREGATE RGB (normalized 0–1 A.U., 0–{args.plot_xmax:g})")
            ws.insert_image(base, col_left, str(p_rgb_agg))
            row = base + stride
            # R
            ws.write(row - 1, col_left, "All-images overlay — R"); ws.insert_image(row, col_left, str(p_rgb_R_all))
            ws.write(row - 1, col_right, "All-images heatmap — R"); ws.insert_image(row, col_right, str(p_rgb_R_map)); row += stride
            # G
            ws.write(row - 1, col_left, "All-images overlay — G"); ws.insert_image(row, col_left, str(p_rgb_G_all))
            ws.write(row - 1, col_right, "All-images heatmap — G"); ws.insert_image(row, col_right, str(p_rgb_G_map)); row += stride
            # B
            ws.write(row - 1, col_left, "All-images overlay — B"); ws.insert_image(row, col_left, str(p_rgb_B_all))
            ws.write(row - 1, col_right, "All-images heatmap — B"); ws.insert_image(row, col_right, str(p_rgb_B_map)); row += stride
            # Per-image overlays
            for name, png in plot_paths_rgb:
                ws.write(row - 1, col_left, f"RGB overlay: {name}"); ws.insert_image(row, col_left, str(png)); row += stride

        # HSV
        ws = writer.sheets["Hist_HSV"]
        if len(hsv_hist_df) > 0:
            base = len(hsv_hist_df) + 4
            ws.write(base - 1, col_left, "AGGREGATE HSV (normalized 0–1 A.U.)")
            ws.insert_image(base, col_left, str(p_hsv_agg))
            row = base + stride
            ws.write(row - 1, col_left, "All-images overlay — H"); ws.insert_image(row, col_left, str(p_hsv_H_all))
            ws.write(row - 1, col_right, "All-images heatmap — H"); ws.insert_image(row, col_right, str(p_hsv_H_map)); row += stride
            ws.write(row - 1, col_left, "All-images overlay — S"); ws.insert_image(row, col_left, str(p_hsv_S_all))
            ws.write(row - 1, col_right, "All-images heatmap — S"); ws.insert_image(row, col_right, str(p_hsv_S_map)); row += stride
            ws.write(row - 1, col_left, "All-images overlay — V"); ws.insert_image(row, col_left, str(p_hsv_V_all))
            ws.write(row - 1, col_right, "All-images heatmap — V"); ws.insert_image(row, col_right, str(p_hsv_V_map)); row += stride
            for name, png in plot_paths_hsv:
                ws.write(row - 1, col_left, f"HSV overlay: {name}"); ws.insert_image(row, col_left, str(png)); row += stride

        # LAB
        ws = writer.sheets["Hist_LAB"]
        if len(lab_hist_df) > 0:
            base = len(lab_hist_df) + 4
            ws.write(base - 1, col_left, "AGGREGATE LAB (normalized 0–1 A.U.)")
            ws.insert_image(base, col_left, str(p_lab_agg))
            row = base + stride
            ws.write(row - 1, col_left, "All-images overlay — L*"); ws.insert_image(row, col_left, str(p_lab_L_all))
            ws.write(row - 1, col_right, "All-images heatmap — L*"); ws.insert_image(row, col_right, str(p_lab_L_map)); row += stride
            ws.write(row - 1, col_left, "All-images overlay — a*"); ws.insert_image(row, col_left, str(p_lab_a_all))
            ws.write(row - 1, col_right, "All-images heatmap — a*"); ws.insert_image(row, col_right, str(p_lab_a_map)); row += stride
            ws.write(row - 1, col_left, "All-images overlay — b*"); ws.insert_image(row, col_left, str(p_lab_b_all))
            ws.write(row - 1, col_right, "All-images heatmap — b*"); ws.insert_image(row, col_right, str(p_lab_b_map)); row += stride
            for name, png in plot_paths_lab:
                ws.write(row - 1, col_left, f"LAB overlay: {name}"); ws.insert_image(row, col_left, str(png)); row += stride

    print(f"Done. Wrote {out_path.name}")

if __name__ == "__main__":
    main()
