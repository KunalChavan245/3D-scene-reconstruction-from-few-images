"""Turn finished runs into results/results.csv and results/RESULTS.md.

    python scripts/aggregate.py
"""

import argparse
from pathlib import Path

import _common  # noqa: F401  (sets up the import path)

from fewview3d.aggregate import collect, report, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-root", default=".")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    rows = collect(Path(args.work_root) / "outputs")
    if not rows:
        raise SystemExit("no results.json files found under outputs/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "results.csv")
    (out / "RESULTS.md").write_text(report(rows))
    print(report(rows))


if __name__ == "__main__":
    main()
