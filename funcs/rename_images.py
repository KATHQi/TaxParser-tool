"""Rename image files in a directory to sequential names starting from 00000000.

Features:
- Dry-run by default. Use --apply to perform renaming.
- Supports common image extensions (jpg/jpeg/png/bmp/tiff/heic/gif). Case-insensitive.
- Non-recursive by default. Use --recursive to process each subdirectory (per-directory renaming starting at --start).
- Uses a safe two-step rename (temporary unique names then final names) to avoid collisions.
- Preserves file extensions.

Examples:
    # dry-run (shows planned renames)
    python rename_images.py --dir F:\\images

    # actually apply renames
    python rename_images.py --dir F:\\images --apply

    # recursive: rename files in every subdirectory (each directory starts from index 0)
    python rename_images.py --dir F:\\images --recursive --apply

Args:
    --dir DIR        Directory containing images to rename
    --apply          If present, perform renaming; otherwise it's a dry-run
    --start N        Starting index (default 0)
    --pad P          Zero-pad width (default 8 -> 00000000)
    --ext extlist    Comma-separated extensions to include (default jpg,jpeg,png,bmp,tiff,gif)
    --sort name|mtime|ctime  Sort order for files (default: name)
    --recursive      Process subdirectories recursively (per-directory renaming)

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid
from typing import List, Tuple
import sys

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}


def _collect_files(directory: Path, exts: List[str], recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return files


def _sort_files(files: List[Path], sort: str) -> List[Path]:
    if sort == 'name':
        return sorted(files, key=lambda p: p.name.lower())
    if sort == 'mtime':
        return sorted(files, key=lambda p: p.stat().st_mtime)
    if sort == 'ctime':
        return sorted(files, key=lambda p: p.stat().st_ctime)
    return files


def _rename_batch(files: List[Path], start: int, pad: int, apply: bool, out_dir: Path | None = None) -> List[Tuple[Path, Path]]:
    """Rename given files in a safe two-step manner.

    Returns list of tuples (src_final_path, dst_final_path) for reporting.
    """
    # Step 1: compute final target names (in same directory as source)
    final_paths = []
    idx = start
    for p in files:
        ext = p.suffix.lower()
        base = f"{idx:0{pad}d}"
        if out_dir:
            dst = out_dir / f"{base}{ext}"
        else:
            dst = p.with_name(f"{base}{ext}")
        final_paths.append(dst)
        idx += 1

    # Step 2: check for potential duplicate final names across this batch
    if len(final_paths) != len(set(map(lambda p: str(p).lower(), final_paths))):
        raise RuntimeError("Duplicate target filenames detected in computed final names")

    # Step 3: temp rename source files to unique temp names in same directory
    temp_paths = []
    for i, src in enumerate(files):
        temp_name = f".renametmp_{uuid.uuid4().hex}_{i}{src.suffix}"
        temp_path = src.with_name(temp_name)
        if apply:
            src.rename(temp_path)
        temp_paths.append(temp_path)

    # Step 4: rename temps to final names
    applied_pairs: List[Tuple[Path, Path]] = []
    for temp, final in zip(temp_paths, final_paths):
        if final.exists():
            # If final exists and wasn't one of the original files being renamed, we avoid overwriting
            # Rollback if already applied? Simpler: raise an error to avoid data loss.
            raise FileExistsError(f"Target file already exists and is not part of this batch: {final}")
        if apply:
            temp.rename(final)
        applied_pairs.append((temp, final))

    return list(zip(files, final_paths))


def rename_images_dir(directory: str, start: int = 0, pad: int = 8, exts: List[str] | None = None,
                      sort: str = 'name', recursive: bool = False, apply: bool = False, out_dir: str | None = None) -> List[Tuple[Path, Path]]:
    p = Path(directory)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(directory)

    exts_set = set([e if e.startswith('.') else f'.{e}' for e in (exts or IMAGE_EXTS)])
    # normalize to lower-case
    exts_set = set(e.lower() for e in exts_set)

    results = []

    if recursive:
        # For recursive mode, rename per-directory to avoid cross-directory collisions
        for d in [p] + [d for d in p.rglob('*') if d.is_dir()]:
            files = _collect_files(d, list(exts_set), recursive=False)
            if not files:
                continue
            files = _sort_files(files, sort)
            pairs = _rename_batch(files, start, pad, apply, out_dir=Path(out_dir) if out_dir else None)
            results.extend(pairs)
    else:
        files = _collect_files(p, list(exts_set), recursive=False)
        files = _sort_files(files, sort)
        pairs = _rename_batch(files, start, pad, apply, out_dir=Path(out_dir) if out_dir else None)
        results.extend(pairs)

    return results


def _print_plan(pairs: List[Tuple[Path, Path]]):
    if not pairs:
        print("No image files found to rename.")
        return
    print("Planned renames:")
    width = max(len(str(src)) for src, _ in pairs)
    for i, (src, dst) in enumerate(pairs):
        print(f"{i:4d}: {str(src):{width}s} -> {str(dst)}")


def main():
    parser = argparse.ArgumentParser(description='Rename image files in a directory to sequential names')
    parser.add_argument('--dir', required=True, help='Directory containing images')
    parser.add_argument('--apply', action='store_true', help='Perform renaming (default is dry-run)')
    parser.add_argument('--start', type=int, default=0, help='Starting index (default 0)')
    parser.add_argument('--pad', type=int, default=8, help='Zero pad width (default 8)')
    parser.add_argument('--ext', default=','.join(sorted([e.lstrip('.') for e in IMAGE_EXTS])), help='Comma-separated extensions to include')
    parser.add_argument('--sort', choices=['name', 'mtime', 'ctime'], default='name', help='Sort order for files')
    parser.add_argument('--recursive', action='store_true', help='Process subdirectories (per-directory renaming)')
    parser.add_argument('--out-dir', help='Optional: place renamed files into this directory (preserve original dirs when --recursive is used)')
    args = parser.parse_args()

    exts = [e.strip().lower() for e in args.ext.split(',') if e.strip()]
    out_dir = args.out_dir

    try:
        pairs = rename_images_dir(args.dir, start=args.start, pad=args.pad, exts=exts, sort=args.sort,
                                  recursive=args.recursive, apply=args.apply, out_dir=out_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

    if args.apply:
        print(f"Renamed {len(pairs)} files.")
    else:
        _print_plan(pairs)
        print(f"Dry-run: {len(pairs)} files would be renamed. Rerun with --apply to perform changes.")


if __name__ == '__main__':
    main()
