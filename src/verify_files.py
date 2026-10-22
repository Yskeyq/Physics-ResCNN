"""Verify a deposited directory against its SHA256SUMS.txt (standard library only)."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    count = 0
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError("Unsafe manifest path")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Checksum mismatch: " + name)
        count += 1
    print(f"Verified {count} files; all SHA-256 checksums match.")


if __name__ == "__main__":
    main()
