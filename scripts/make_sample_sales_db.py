"""Create a small SQLite database to try the Data Reports page.

    python scripts/make_sample_sales_db.py data/reports

Writes <folder>/sales.sqlite3 with one table, `sales` (120 orders across 4 regions and
4 products over 2026-01..2026-06). The data is synthetic and contains no personal
information. Point HAI_DATABASE_ROOT at the folder, then connect "sales.sqlite3" in the UI.
Refuses to overwrite an existing file.
"""
from __future__ import annotations

import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

REGIONS = ["west", "east", "north", "south"]
PRODUCTS = ["widget", "gadget", "gizmo", "doohickey"]
PRICES = {"widget": 24.5, "gadget": 61.0, "gizmo": 12.25, "doohickey": 99.9}


def build(folder: str | Path, rows: int = 120, seed: int = 7) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "sales.sqlite3"
    if path.exists():
        raise FileExistsError(f"{path} already exists; delete it first to recreate")
    rng = random.Random(seed)
    start = date(2026, 1, 1)
    records = []
    for order_id in range(1, rows + 1):
        product = rng.choice(PRODUCTS)
        quantity = rng.randint(1, 9)
        records.append((
            order_id, rng.choice(REGIONS), product, quantity,
            round(PRICES[product] * quantity, 2),
            (start + timedelta(days=rng.randint(0, 180))).isoformat()))
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT NOT NULL, "
                   "product TEXT NOT NULL, quantity INTEGER NOT NULL, amount REAL NOT NULL, "
                   "ordered_on DATE NOT NULL)")
        db.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", records)
    return path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        path = build(argv[1])
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Created {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
