#!/usr/bin/env python3
"""
Microstage EDF: Extended Depth of Field with optional illumination correction
and OME-TIFF export.

Color policy:
 - Internal: BGR (OpenCV default).
 - Gray: cv2.COLOR_BGR2GRAY.
 - PNG/TIFF via OpenCV: write as-is (BGR or gray).
 - OME-TIFF via tifffile: convert BGR->RGB at write time.

Color-preserving fusion:
 - rgb_mix      : (legacy) fuse all RGB/BGR channels directly (can desaturate).
 - lab_l_only   : fuse only L in CIE Lab; take a/b from depth-winner slice.
 - hsv_v_only   : fuse only V in HSV; take H/S from depth-winner slice.

For best result
python microstage_edf.py --illum stack_median --illum-sigma 60 --focus tenengrad --save-ome --px 0.325 --py 0.325 --color-mode lab_l_only

Outputs named "<folder>-deep.*" as requested.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import tifffile as tiff

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# --------------------- File discovery & sorting ---------------------
def find_stack_images(folder: Path, pattern: Optional[str]) -> List[Path]:
    if pattern:
        candidates = list(folder.glob(pattern))
    else:
        candidates = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    def key(p: Path):
        m = re.search(r"(\d{4,})", p.stem)
        primary = int(m.group(1)) if m else float("inf")
        return (primary, p.name.lower())
    return sorted([p for p in candidates if p.is_file()], key=key)

# --------------------------- I/O utilities --------------------------
def read_image(path: Path) -> np.ndarray:
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise RuntimeError(f"Failed to read {path}")
    if data.ndim == 2:
        return data
    if data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data  # BGR

def ensure_same_channels(images: List[np.ndarray]) -> bool:
    chs = [(1 if im.ndim == 2 else im.shape[2]) for im in images]
    return len(set(chs)) == 1

def harmonize_sizes(images: List[np.ndarray]) -> List[np.ndarray]:
    h_min = min(im.shape[0] for im in images)
    w_min = min(im.shape[1] for im in images)
    return [cv2.resize(im, (w_min, h_min), interpolation=cv2.INTER_AREA) for im in images]

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)

# ------------------------- Illumination correction ------------------
def estimate_flatfield_stack_median(stack: List[np.ndarray], blur_sigma: float = 50.0) -> np.ndarray:
    arr = np.stack(stack, axis=0).astype(np.float32)
    if arr.ndim == 4:
        med = np.median(arr, axis=0)
        for c in range(med.shape[2]):
            med[..., c] = cv2.GaussianBlur(med[..., c], (0, 0), blur_sigma)
    else:
        med = np.median(arr, axis=0)
        med = cv2.GaussianBlur(med, (0, 0), blur_sigma)
    return np.clip(med, 1e-6, None)

def estimate_flatfield_per_slice(img: np.ndarray, blur_sigma: float = 50.0) -> np.ndarray:
    if img.ndim == 3:
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            out[..., c] = cv2.GaussianBlur(img[..., c].astype(np.float32), (0, 0), blur_sigma)
    else:
        out = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur_sigma)
    return np.clip(out, 1e-6, None)

def apply_flatfield(img: np.ndarray, flat: np.ndarray) -> np.ndarray:
    imgf = img.astype(np.float32)
    if imgf.ndim == 2 and flat.ndim == 3:
        flat = cv2.cvtColor(flat.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if imgf.ndim == 3 and flat.ndim == 2:
        flat = np.repeat(flat[..., None], imgf.shape[2], axis=2)
    mean_flat = float(np.mean(flat))
    corrected = imgf * (mean_flat / flat)
    return np.clip(corrected, 0, 255).astype(img.dtype)

# --------------------------- Focus measures -------------------------
def fm_tenengrad(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return gx*gx + gy*gy

def fm_variance(gray: np.ndarray, win: int = 7) -> np.ndarray:
    mu = cv2.GaussianBlur(gray, (0, 0), win/6.0)
    mu2 = cv2.GaussianBlur(gray*gray, (0, 0), win/6.0)
    return np.maximum(mu2 - mu*mu, 0)

def fm_laplacian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return lap*lap

def fm_brenner(gray: np.ndarray) -> np.ndarray:
    dx = gray[:, 2:] - gray[:, :-2]
    dy = gray[2:, :] - gray[:-2, :]
    m = np.zeros_like(gray)
    m[:, 1:-1] += 0.5*(dx[:, :-1]**2 + dx[:, 1:]**2)
    m[1:-1, :] += 0.5*(dy[:-1, :]**2 + dy[1:, :]**2)
    return m

def fm_vollath_f4(gray: np.ndarray) -> np.ndarray:
    s1 = gray[1:, :] * gray[:-1, :]
    s2 = gray[2:, :] * gray[:-2, :]
    m = np.zeros_like(gray)
    m[1:, :] += s1
    m[2:, :] -= s2
    return np.maximum(m, 0)

FOCUS_FUNCS = {
    "tenengrad": fm_tenengrad,
    "variance":  fm_variance,
    "laplacian": fm_laplacian,
    "brenner":   fm_brenner,
    "vollath_f4": fm_vollath_f4,
}

# -------------------------- Multiscale fusion -----------------------
def normalize_weights(weight_maps: List[np.ndarray], blur_sigma: float = 1.0, eps: float = 1e-12) -> List[np.ndarray]:
    W = np.stack(weight_maps, axis=0)  # [N,H,W]
    W = np.maximum(W, 0)
    for i in range(W.shape[0]):
        W[i] = cv2.GaussianBlur(W[i], (0, 0), blur_sigma)
    denom = np.sum(W, axis=0, keepdims=True) + eps
    Wn = W / denom
    return [Wn[i] for i in range(Wn.shape[0])]

def build_gaussian_pyr(img: np.ndarray, levels: int) -> List[np.ndarray]:
    G = [img]
    for _ in range(1, levels):
        G.append(cv2.pyrDown(G[-1]))
    return G

def build_laplacian_pyr(img: np.ndarray, levels: int) -> List[np.ndarray]:
    G = build_gaussian_pyr(img, levels)
    L = []
    for i in range(levels-1):
        up = cv2.pyrUp(G[i+1], dstsize=(G[i].shape[1], G[i].shape[0]))
        L.append(G[i].astype(np.float32) - up.astype(np.float32))
    L.append(G[-1].astype(np.float32))
    return L

def reconstruct_from_laplacian(L: List[np.ndarray]) -> np.ndarray:
    img = L[-1]
    for i in range(len(L)-2, -1, -1):
        up = cv2.pyrUp(img, dstsize=(L[i].shape[1], L[i].shape[0]))
        img = up + L[i]
    return img

# -------------------------- Fusion wrappers -------------------------
def compute_focus_and_weights(images: List[np.ndarray], focus: str, sobel_ksize: int) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    grays = [to_gray(im) for im in images]
    if focus == "tenengrad":
        fmaps = [FOCUS_FUNCS[focus](g, ksize=sobel_ksize) for g in grays]
    elif focus in ("laplacian",):
        fmaps = [FOCUS_FUNCS[focus](g, ksize=3) for g in grays]
    else:
        fmaps = [FOCUS_FUNCS[focus](g) for g in grays]
    depth_idx = np.argmax(np.stack(fmaps, axis=0), axis=0).astype(np.uint16)
    weights = normalize_weights(fmaps, blur_sigma=1.0)
    return fmaps, depth_idx, weights

def fuse_rgb_mix(images: List[np.ndarray], weights: List[np.ndarray], levels: int) -> np.ndarray:
    L_pyrs = [build_laplacian_pyr(im.astype(np.float32), levels) for im in images]
    W_pyrs = [build_gaussian_pyr(w, levels) for w in weights]
    is_gray = images[0].ndim == 2
    fused_pyr = []
    for lvl in range(levels):
        acc = np.zeros_like(L_pyrs[0][lvl], dtype=np.float32)
        for s in range(len(images)):
            w = W_pyrs[s][lvl]
            if not is_gray:
                w = w[..., None]
            acc += L_pyrs[s][lvl] * w
        fused_pyr.append(acc)
    fused = reconstruct_from_laplacian(fused_pyr)
    return np.clip(fused, 0, 255).astype(np.uint8)

def fuse_lab_l_only(images: List[np.ndarray], weights: List[np.ndarray], depth_idx: np.ndarray, levels: int) -> np.ndarray:
    # Convert to Lab
    labs = [cv2.cvtColor(im, cv2.COLOR_BGR2Lab).astype(np.float32) for im in images]
    Ls   = [lab[..., 0] for lab in labs]
    # Fuse L (single-channel) with pyramids
    L_pyrs = [build_laplacian_pyr(L.astype(np.float32), levels) for L in Ls]
    W_pyrs = [build_gaussian_pyr(w, levels) for w in weights]
    fused_L_pyr = []
    for lvl in range(levels):
        acc = np.zeros_like(L_pyrs[0][lvl], dtype=np.float32)
        for s in range(len(images)):
            acc += L_pyrs[s][lvl] * W_pyrs[s][lvl]
        fused_L_pyr.append(acc)
    fused_L = reconstruct_from_laplacian(fused_L_pyr)
    fused_L = np.clip(fused_L, 0, 255).astype(np.uint8)
    # Take a,b from depth-winner slice
    H, W = depth_idx.shape
    ab = np.zeros((H, W, 2), dtype=np.uint8)
    # Gather per-pixel a,b from the winning Lab image
    stack_a = np.stack([lab[...,1] for lab in labs], axis=0)  # [N,H,W]
    stack_b = np.stack([lab[...,2] for lab in labs], axis=0)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    ab[...,0] = stack_a[depth_idx, rows, cols]
    ab[...,1] = stack_b[depth_idx, rows, cols]
    fused_lab = np.dstack([fused_L, ab[...,0], ab[...,1]]).astype(np.uint8)
    # Back to BGR
    bgr = cv2.cvtColor(fused_lab, cv2.COLOR_Lab2BGR)
    return bgr

def fuse_hsv_v_only(images: List[np.ndarray], weights: List[np.ndarray], depth_idx: np.ndarray, levels: int) -> np.ndarray:
    hsvs = [cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32) for im in images]
    Vs   = [hsv[..., 2] for hsv in hsvs]
    # Fuse V channel
    V_pyrs = [build_laplacian_pyr(V.astype(np.float32), levels) for V in Vs]
    W_pyrs = [build_gaussian_pyr(w, levels) for w in weights]
    fused_V_pyr = []
    for lvl in range(levels):
        acc = np.zeros_like(V_pyrs[0][lvl], dtype=np.float32)
        for s in range(len(images)):
            acc += V_pyrs[s][lvl] * W_pyrs[s][lvl]
        fused_V_pyr.append(acc)
    fused_V = reconstruct_from_laplacian(fused_V_pyr)
    fused_V = np.clip(fused_V, 0, 255).astype(np.uint8)
    # Take H,S from winner
    H, W = depth_idx.shape
    stack_H = np.stack([hsv[...,0] for hsv in hsvs], axis=0)
    stack_S = np.stack([hsv[...,1] for hsv in hsvs], axis=0)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    Hwin = stack_H[depth_idx, rows, cols].astype(np.uint8)
    Swin = stack_S[depth_idx, rows, cols].astype(np.uint8)
    fused_hsv = np.dstack([Hwin, Swin, fused_V]).astype(np.uint8)
    bgr = cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2BGR)
    return bgr

def fuse_stack(images: List[np.ndarray], color_mode: str, focus: str, levels: int, sobel_ksize: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (fused_img uint8 BGR, depth_index uint16)
    """
    fmaps, depth_idx, weights = compute_focus_and_weights(images, focus, sobel_ksize)
    if images[0].ndim == 2:  # grayscale stack
        # reuse rgb_mix on single channel
        fused = fuse_rgb_mix(images, weights, levels)
        return fused, depth_idx

    if color_mode == "rgb_mix":
        fused = fuse_rgb_mix(images, weights, levels)
    elif color_mode == "lab_l_only":
        fused = fuse_lab_l_only(images, weights, depth_idx, levels)
    elif color_mode == "hsv_v_only":
        fused = fuse_hsv_v_only(images, weights, depth_idx, levels)
    else:
        raise ValueError("Unknown color_mode")
    return fused, depth_idx

# ------------------------------ Main --------------------------------
def main():
    ap = argparse.ArgumentParser(description="EDF (focus stacking) with illumination correction, color-preserving fusion, and OME-TIFF export.")
    ap.add_argument("--pattern", type=str, default=None, help="Glob filter, e.g. 'stack_*.tif'. If omitted, use all common image types.")
    ap.add_argument("--focus", type=str, default="tenengrad", choices=list(FOCUS_FUNCS.keys()), help="Focus measure.")
    ap.add_argument("--levels", type=int, default=5, help="Pyramid levels (3–6 typical).")
    ap.add_argument("--sobel-ksize", type=int, default=3, choices=[3,5,7], help="Sobel kernel (Tenengrad).")
    ap.add_argument("--illum", type=str, default="none", choices=["none","per_slice","stack_median"], help="Illumination correction mode.")
    ap.add_argument("--illum-sigma", type=float, default=50.0, help="Gaussian sigma for flat-field estimation.")
    ap.add_argument("--color-mode", type=str, default="lab_l_only", choices=["rgb_mix","lab_l_only","hsv_v_only"], help="Color handling for fusion.")
    ap.add_argument("--px", type=float, default=None, help="Physical pixel size X (micrometers).")
    ap.add_argument("--py", type=float, default=None, help="Physical pixel size Y (micrometers).")
    ap.add_argument("--save-ome", action="store_true", help="Save fused/depth as OME-TIFF.")
    args = ap.parse_args()

    folder = Path(__file__).resolve().parent
    base_name = f"{folder.name}-deep"

    files = find_stack_images(folder, args.pattern)
    files = [f for f in files if f.suffix.lower() in IMG_EXTS]
    if len(files) < 2:
        print("Need at least two images in this folder.", file=sys.stderr)
        for f in files:
            print(" -", f.name)
        sys.exit(2)

    print(f"Found {len(files)} images:")
    for f in files:
        print(" -", f.name)

    images = [read_image(p) for p in files]
    if not ensure_same_channels(images):
        raise ValueError("Mixed channel counts in stack (e.g., grayscale + BGR). Make them consistent.")
    images = harmonize_sizes(images)

    # Illumination correction
    if args.illum == "stack_median":
        flat = estimate_flatfield_stack_median(images, blur_sigma=args.illum_sigma)
        images = [apply_flatfield(im, flat) for im in images]
    elif args.illum == "per_slice":
        images = [apply_flatfield(im, estimate_flatfield_per_slice(im, blur_sigma=args.illum_sigma)) for im in images]

    fused, depth = fuse_stack(images, color_mode=args.color_mode, focus=args.focus, levels=args.levels, sobel_ksize=args.sobel_ksize)

    # Save PNG/TIFF with naming convention (BGR direct)
    out_img = folder / f"{base_name}.png"
    out_dep = folder / f"{base_name}_depth.tiff"
    cv2.imwrite(str(out_img), fused)
    cv2.imwrite(str(out_dep), depth)
    print(f"Saved:\n  {out_img}\n  {out_dep}")

    # Optional OME-TIFF
    if args.save_ome:
        is_color = (fused.ndim == 3 and fused.shape[2] == 3)
        meta = {"axes": "YXC" if is_color else "YX"}
        if args.px is not None:
            meta["PhysicalSizeX"] = float(args.px); meta["PhysicalSizeXUnit"] = "µm"
        if args.py is not None:
            meta["PhysicalSizeY"] = float(args.py); meta["PhysicalSizeYUnit"] = "µm"

        ome_fused = folder / f"{base_name}.ome.tif"
        if is_color:
            rgb = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)
            tiff.imwrite(str(ome_fused), rgb, photometric="rgb", metadata=meta, bigtiff=False)
        else:
            tiff.imwrite(str(ome_fused), fused, photometric="minisblack", metadata=meta, bigtiff=False)

        meta_d = {"axes": "YX"}
        if args.px is not None:
            meta_d["PhysicalSizeX"] = float(args.px); meta_d["PhysicalSizeXUnit"] = "µm"
        if args.py is not None:
            meta_d["PhysicalSizeY"] = float(args.py); meta_d["PhysicalSizeYUnit"] = "µm"
        ome_depth = folder / f"{base_name}_depth.ome.tif"
        tiff.imwrite(str(ome_depth), depth, photometric="minisblack", metadata=meta_d, bigtiff=False)

        print(f"Saved OME-TIFF:\n  {ome_fused}\n  {ome_depth}")

    print("Done.")

if __name__ == "__main__":
    main()
