"""Create a deterministic sample stock-price CSV for local testing."""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate(path: Path, days: int = 360) -> None:
    random.seed(7)
    start = datetime(2024, 1, 2)
    price = 142.0
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for day in range(days):
            current_date = start + timedelta(days=day)
            trend = 0.045
            seasonality = math.sin(day / 17) * 0.55
            shock = random.gauss(0, 1.15)
            price = max(5, price + trend + seasonality + shock)
            open_price = price + random.gauss(0, 0.8)
            high = max(open_price, price) + abs(random.gauss(0, 1.2))
            low = min(open_price, price) - abs(random.gauss(0, 1.2))
            volume = int(1_800_000 + random.random() * 550_000 + day * 900)
            writer.writerow(
                [
                    current_date.date().isoformat(),
                    f"{open_price:.2f}",
                    f"{high:.2f}",
                    f"{low:.2f}",
                    f"{price:.2f}",
                    volume,
                ]
            )


def main() -> None:
    generate(Path("data/sample_stock_prices.csv"))
    print("Created data/sample_stock_prices.csv")


if __name__ == "__main__":
    main()
