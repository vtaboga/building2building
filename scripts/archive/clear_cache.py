#!/usr/bin/env python3
"""Clear the B2B derivation store cache.

Usage:
    python scripts/clear_cache.py                  # interactive confirmation
    python scripts/clear_cache.py --yes             # skip confirmation
    python scripts/clear_cache.py --pattern '*controllable*'  # only matching dirs
    python scripts/clear_cache.py --dry-run         # show what would be deleted
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

from building2building.env import store_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clear the B2B store cache.")
    p.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt."
    )
    p.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print what would be deleted without actually deleting.",
    )
    p.add_argument(
        "--pattern",
        "-p",
        type=str,
        default=None,
        help="Only delete entries whose directory name matches this glob pattern "
        "(e.g. '*controllable*').",
    )
    return p.parse_args()


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    args = _parse_args()
    cache_dir = store_path()

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        sys.exit(0)

    if args.pattern:
        entries = sorted(
            e
            for e in cache_dir.iterdir()
            if e.is_dir() and fnmatch.fnmatch(e.name, args.pattern)
        )
    else:
        entries = sorted(e for e in cache_dir.iterdir() if e.is_dir())

    if not entries:
        label = f" matching '{args.pattern}'" if args.pattern else ""
        print(f"No cache entries{label} found in {cache_dir}")
        sys.exit(0)

    total_size = sum(_dir_size(e) for e in entries)
    print(f"Cache directory: {cache_dir}")
    print(f"Entries to delete: {len(entries)}  ({_human_size(total_size)})")
    if args.pattern:
        print(f"Pattern filter: {args.pattern}")
    print()

    if args.dry_run:
        for e in entries:
            print(f"  [dry-run] would delete {e.name}")
        sys.exit(0)

    if not args.yes:
        answer = input("Delete these entries? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            sys.exit(1)

    deleted = 0
    for e in entries:
        shutil.rmtree(e)
        deleted += 1

    print(f"Deleted {deleted} cache entries ({_human_size(total_size)}).")


if __name__ == "__main__":
    main()
