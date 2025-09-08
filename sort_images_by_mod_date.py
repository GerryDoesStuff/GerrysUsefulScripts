#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sort_images_by_mod_date.py
#
# Default behavior:
#   Process EACH immediate subdirectory (one level down), non-recursively
#   within each, creating S1..S* inside each subdirectory.
#
# Optional:
#   Use --here to process ONLY the current folder (non-recursive).
#
# Grouping behavior:
# - Sort by mtime ascending; tie-break by name.
# - Partition into groups of N (default 90; configurable via --group-size / -n).
# - For each position j in a group, move the j-th item of every group to S{j+1}
#   and rename as "{r}fr_fr{r}{ext}" where r is the group index (0,1,2,...).
#
# Subdir group-size auto-detection:
# - When processing subdirectories, we try to extract N from the subfolder's
#   name using a hyphen followed by optional spaces and digits, e.g. "- 9" or "-9".
#   Regex (shown textually): r"-\s*(\d+)"
# - If found, that number is used as the group size for that subfolder.
# - If not found, we fall back to the provided -n/--group-size value (default 90).
# - Use --no-derive to disable name-based detection.
#
# Windows examples:
#   py sort_images_by_mod_date.py                # process all immediate subfolders (default)
#   py sort_images_by_mod_date.py -n 12         # same, fallback N=12 where no number is found
#   py sort_images_by_mod_date.py --no-derive -n 9
#   py sort_images_by_mod_date.py --here -n 9   # process only the current folder
#
from pathlib import Path
import shutil
import argparse
import sys
import re
from typing import Tuple, Optional

# Accepted image extensions (lowercase)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}

DEFAULT_GROUP_SIZE = 90  # images per group if not overridden
SUBDIR_SKIP_PATTERN = re.compile(r"^S\d+$")  # skip S1, S2, ... directories
DERIVE_REGEX = re.compile(r"-\s*(\d+)")      # capture digits following '-' and optional spaces

def chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def safe_move_with_rename(src: Path, dest_dir: Path, new_name: str) -> Path:
    '''Move src to dest_dir/new_name; on collision, append -1, -2, ... to the stem.'''
    dest_dir.mkdir(exist_ok=True)
    dest_path = dest_dir / new_name
    if not dest_path.exists():
        shutil.move(str(src), str(dest_path))
        return dest_path

    stem, suffix = dest_path.stem, dest_path.suffix
    k = 1
    while True:
        cand = dest_dir / f"{stem}-{k}{suffix}"
        if not cand.exists():
            shutil.move(str(src), str(cand))
            return cand
        k += 1

def derive_group_size_from_name(dirname: str) -> Optional[int]:
    r'''Return the first integer matched by the pattern "-\s*(\d+)" in a directory name.'''
    m = DERIVE_REGEX.search(dirname)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def process_one_directory(base: Path, group_size: int) -> Tuple[int, int, int]:
    '''Process a single directory (non-recursive). Return (num_images, num_groups, moved_total).'''
    images = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        return (0, 0, 0)

    # Sort by mtime (earliest first), tie-break by name for stability
    images.sort(key=lambda p: (p.stat().st_mtime, p.name))

    # Partition into groups of group_size by sorted order
    groups = list(chunk(images, group_size))
    group_count = len(groups)
    if group_count == 0:
        return (0, 0, 0)

    # Max width among groups determines how many S-folders we might need
    max_group_len = min(group_size, len(images))

    # Move images at position j from each group to S{j+1}, renaming as specified
    moved_total = 0
    for j in range(max_group_len):
        s_dir = base / f"S{j+1}"
        for r, g in enumerate(groups):
            if j >= len(g):
                continue  # this group doesn't have a j-th image
            src = g[j]
            new_name = f"{r}fr_fr{r}{src.suffix}"
            safe_move_with_rename(src, s_dir, new_name)
            moved_total += 1

    return (len(images), group_count, moved_total)

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Sort images by mtime into S1..Sk folders. By default, process each immediate "
            "subdirectory; use --here to process only the current folder. Optionally derive "
            "group size from subdirectory names."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "-n", "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help="Number of images per group before interleaving (must be >= 1)."
    )
    p.add_argument(
        "--no-derive",
        action="store_true",
        help="Disable deriving group size from subdirectory names when processing subdirectories."
    )
    p.add_argument(
        "--here",
        action="store_true",
        help="Process ONLY the current folder (non-recursive). Without this flag, subdirectories are processed."
    )
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    fallback_group_size = args.group_size

    if fallback_group_size < 1:
        print(f"Invalid --group-size: {fallback_group_size}. It must be >= 1.", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent

    if args.here:
        # Single-directory mode
        imgs, groups, moved = process_one_directory(root, fallback_group_size)
        if imgs == 0:
            print("No images found in the current folder. (Try without --here to process subfolders.)")
        else:
            max_pos = min(fallback_group_size, imgs)
            print(f"[{root.name}] Group size: {fallback_group_size}")
            print(f"Processed {imgs} image(s) across {groups} group(s). "
                  f"Moved {moved} image(s) into S1..S{max_pos}.")
        return 0

    # Default: process immediate subdirectories
    any_dir = False
    total_imgs = total_groups = total_moved = 0

    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        # Skip S1, S2, ... directories at parent level
        if SUBDIR_SKIP_PATTERN.match(d.name):
            continue

        any_dir = True

        # Determine group size for this subdir
        if args.no_derive:
            gs = fallback_group_size
            reason = "forced"
        else:
            derived = derive_group_size_from_name(d.name)
            if derived and derived >= 1:
                gs = derived
                reason = "derived"
            else:
                gs = fallback_group_size
                reason = "fallback"

        imgs, groups, moved = process_one_directory(d, gs)
        total_imgs += imgs
        total_groups += groups
        total_moved += moved

        if imgs == 0:
            print(f"[{d.name}] No images found. (group-size {gs}, {reason})")
        else:
            max_pos = min(gs, imgs)
            print(f"[{d.name}] Group size: {gs} ({reason})")
            print(f"Processed {imgs} image(s) across {groups} group(s). "
                  f"Moved {moved} image(s) into S1..S{max_pos}.")

    if not any_dir:
        print("No subdirectories found to process in the current location.")
    else:
        print(f"\nTOTAL — images: {total_imgs}, groups: {total_groups}, moved: {total_moved}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
