#!/usr/bin/env python3
"""
Join photos in the script's folder into a 1 fps video, ordered by timestamp in filename.

Expected timestamp pattern somewhere in filename:
    hh-mm-ss dd-mm-yyyy
Example:
    14-03-27 05-01-2026.jpg

Adds overlay text "Layer [x]" to the upper-right corner of each frame, where x is the
0-based index in the sorted frame list.

Default output codec is mp4v to avoid OpenH264/H.264 dependency issues on Windows.

Requires:
    pip install opencv-python
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2


TS_RE = re.compile(
    r"(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})\s+(?P<d>\d{2})-(?P<mo>\d{2})-(?P<y>\d{4})"
)


@dataclass(frozen=True)
class FrameItem:
    path: Path
    ts: datetime


def parse_timestamp_from_name(name: str) -> Optional[datetime]:
    m = TS_RE.search(name)
    if not m:
        return None
    try:
        return datetime(
            int(m.group("y")),
            int(m.group("mo")),
            int(m.group("d")),
            int(m.group("h")),
            int(m.group("m")),
            int(m.group("s")),
        )
    except ValueError:
        return None


def iter_images(folder: Path, exts: Iterable[str]) -> List[FrameItem]:
    items: List[FrameItem] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        ts = parse_timestamp_from_name(p.name)
        if ts is None:
            continue
        items.append(FrameItem(path=p, ts=ts))
    items.sort(key=lambda x: (x.ts, x.path.name.lower()))
    return items


def read_image_bgr(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def resize_and_pad(img, target_w: int, target_h: int):
    """
    Resize to fit within target dimensions while preserving aspect ratio,
    then pad with black to exact size.
    """
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = (target_h - new_h) - top
    left = (target_w - new_w) // 2
    right = (target_w - new_w) - left

    canvas = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return canvas


def ensure_even(x: int) -> int:
    # Some encoders require even dimensions
    return x if x % 2 == 0 else x - 1


def draw_layer_label(frame, layer_index: int, margin: int = 16) -> None:
    """
    Draw "Layer [x]" in the upper-right corner. Uses a black outline for readability.
    """
    text = f"Layer {layer_index}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = frame.shape[:2]

    # Anchor at upper-right with margin
    x = max(margin, w - margin - tw)
    y = max(margin + th, margin + th)  # ensure visible

    # Outline (black) then fill (white)
    outline_thickness = thickness + 2
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Create a 1 fps video from timestamped photos in this folder.")
    ap.add_argument("--out", default="output.mp4", help="Output video filename (default: output.mp4)")
    ap.add_argument("--fps", type=float, default=1.0, help="Frames per second (default: 1.0)")
    ap.add_argument(
        "--exts",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp",
        help="Comma-separated allowed extensions (default: common image types)",
    )
    ap.add_argument(
        "--size",
        default="",
        help="Optional fixed size WxH, e.g. 1920x1080. If omitted, uses first image size (made even).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any image cannot be read; otherwise skip unreadable images.",
    )
    args = ap.parse_args()

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    frames = iter_images(script_dir, exts)

    if not frames:
        print(
            f"No images with timestamp pattern 'hh-mm-ss dd-mm-yyyy' found in: {script_dir}\n"
            f"Allowed extensions: {', '.join(sorted(exts))}",
            file=sys.stderr,
        )
        return 2

    # Determine output size
    if args.size:
        m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", args.size.strip().lower())
        if not m:
            print("Invalid --size. Use WxH, e.g. 1920x1080", file=sys.stderr)
            return 2
        target_w, target_h = int(m.group(1)), int(m.group(2))
    else:
        first = read_image_bgr(frames[0].path)
        if first is None:
            print(f"Cannot read first image: {frames[0].path}", file=sys.stderr)
            return 2
        target_h, target_w = first.shape[:2]

    target_w = ensure_even(int(target_w))
    target_h = ensure_even(int(target_h))
    if target_w <= 0 or target_h <= 0:
        print("Invalid output dimensions after even-adjustment.", file=sys.stderr)
        return 2

    out_path = (script_dir / args.out).resolve()

    # Avoid H.264 to prevent OpenH264/FFmpeg DLL issues on Windows.
    # Try mp4v first; fall back to MJPG if needed.
    fourcc_candidates = ["mp4v", "MJPG"]

    writer = None
    used_fourcc = None
    for fcc in fourcc_candidates:
        fourcc = cv2.VideoWriter_fourcc(*fcc)
        w = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (target_w, target_h))
        if w.isOpened():
            writer = w
            used_fourcc = fcc
            break

    if writer is None:
        print("Failed to open a VideoWriter (tried mp4v, MJPG).", file=sys.stderr)
        return 3

    print(f"Input folder: {script_dir}")
    print(f"Frames found: {len(frames)}")
    print(f"Output: {out_path}  | fps={args.fps}  | size={target_w}x{target_h}  | codec={used_fourcc}")

    written = 0
    skipped = 0
    for idx, item in enumerate(frames):
        i = idx + 1
        img = read_image_bgr(item.path)
        if img is None:
            msg = f"[{i}/{len(frames)}] Unreadable image: {item.path.name}"
            if args.strict:
                print(msg, file=sys.stderr)
                writer.release()
                return 4
            print(msg + " (skipped)", file=sys.stderr)
            skipped += 1
            continue

        frame = resize_and_pad(img, target_w, target_h)
        if frame is None:
            msg = f"[{i}/{len(frames)}] Invalid image dimensions: {item.path.name}"
            if args.strict:
                print(msg, file=sys.stderr)
                writer.release()
                return 4
            print(msg + " (skipped)", file=sys.stderr)
            skipped += 1
            continue

        draw_layer_label(frame, layer_index=idx, margin=16)

        writer.write(frame)
        written += 1

    writer.release()

    if written == 0:
        print("No frames written (all images skipped).", file=sys.stderr)
        return 5

    print(f"Done. Written frames: {written}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
