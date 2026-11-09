"""CLI: refresh the ICD-11 definition cache against the local container. Run this
occasionally (whenever the synthetic dataset's code set changes) rather than keeping the
container running continuously — see docs/environment.md.

Usage:
    docker compose up -d
    python -m src.icd11.refresh_cache --codes-file data/processed/icd11_codes_used.txt \\
        --cache-path artifacts/icd11_cache.json --base-url http://localhost:8081
    docker compose down
"""
import argparse
import sys

from src.icd11.cache import ICD11Cache
from src.icd11.client import ICD11APIClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-file", required=True,
                         help="Text file, one ICD-11 code per line.")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--base-url", default="http://localhost:8081")
    parser.add_argument("--release", default="2026-01")
    args = parser.parse_args()

    codes = [line.strip() for line in open(args.codes_file, encoding="utf-8") if line.strip()]
    client = ICD11APIClient(base_url=args.base_url, release=args.release)

    if not client.health_check():
        print(f"ERROR: ICD-11 API not reachable at {args.base_url}. "
              f"Is `docker compose up -d` running?", file=sys.stderr)
        sys.exit(1)

    cache = ICD11Cache(args.cache_path)
    errors = cache.bulk_prefetch(codes, client)

    print(f"Cache now holds {len(cache)} definitions ({len(codes)} requested).")
    if errors:
        print(f"{len(errors)} codes failed to fetch:", file=sys.stderr)
        for code, error in errors.items():
            print(f"  {code}: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
