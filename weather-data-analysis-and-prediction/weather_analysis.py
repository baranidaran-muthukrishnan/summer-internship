#!/usr/bin/env python3
"""Analyze daily temperatures and create a simple linear-regression forecast.

Input CSV must contain a `date` column (YYYY-MM-DD) and one temperature column:
`temperature`, `temp_c`, `temperature_c`, or `tavg`.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev


TEMPERATURE_COLUMNS = ("temperature", "temp_c", "temperature_c", "tavg")


def load_observations(path: Path) -> list[tuple[date, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            raise ValueError("CSV needs a 'date' column in YYYY-MM-DD format.")
        temp_column = next((c for c in TEMPERATURE_COLUMNS if c in reader.fieldnames), None)
        if not temp_column:
            accepted = ", ".join(TEMPERATURE_COLUMNS)
            raise ValueError(f"CSV needs one temperature column: {accepted}.")
        rows = []
        for line, row in enumerate(reader, start=2):
            try:
                rows.append((datetime.strptime(row["date"].strip(), "%Y-%m-%d").date(), float(row[temp_column])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid date or temperature at CSV line {line}.") from exc
    rows.sort(key=lambda item: item[0])
    if len(rows) < 14:
        raise ValueError("At least 14 daily observations are required.")
    return rows


def demo_observations() -> list[tuple[date, float]]:
    """Deterministic illustrative data; not real weather observations."""
    start = date.today() - timedelta(days=364)
    rng = random.Random(17)
    return [
        (start + timedelta(days=day), round(25 + 6 * math.sin((day - 35) * 2 * math.pi / 365) + 0.006 * day + rng.gauss(0, 1.2), 1))
        for day in range(365)
    ]


def linear_regression(rows: list[tuple[date, float]]) -> tuple[float, float, float]:
    x = [(day - rows[0][0]).days for day, _ in rows]
    y = [temperature for _, temperature in rows]
    x_bar, y_bar = mean(x), mean(y)
    denominator = sum((value - x_bar) ** 2 for value in x)
    slope = sum((a - x_bar) * (b - y_bar) for a, b in zip(x, y)) / denominator
    intercept = y_bar - slope * x_bar
    predictions = [intercept + slope * value for value in x]
    ss_total = sum((value - y_bar) ** 2 for value in y)
    ss_residual = sum((actual - predicted) ** 2 for actual, predicted in zip(y, predictions))
    r_squared = 1 - ss_residual / ss_total if ss_total else 1.0
    return slope, intercept, r_squared


def make_chart(rows: list[tuple[date, float]], forecasts: list[tuple[date, float]], output: Path) -> None:
    width, height, pad = 1100, 540, 70
    all_rows = rows + forecasts
    start, end = all_rows[0][0], all_rows[-1][0]
    values = [temp for _, temp in all_rows]
    ymin, ymax = math.floor(min(values) - 2), math.ceil(max(values) + 2)
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    def point(day: date, temp: float) -> tuple[float, float]:
        x = pad + (day - start).days / max((end - start).days, 1) * plot_w
        y = height - pad - (temp - ymin) / max(ymax - ymin, 1) * plot_h
        return x, y
    def path(data: list[tuple[date, float]]) -> str:
        return " ".join(("M" if i == 0 else "L") + f" {point(d, t)[0]:.1f},{point(d, t)[1]:.1f}" for i, (d, t) in enumerate(data))
    grid = []
    for tick in range(ymin, ymax + 1, max(1, math.ceil((ymax - ymin) / 6))):
        _, y = point(start, tick)
        grid.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" class="grid"/><text x="{pad-12}" y="{y+4:.1f}" text-anchor="end">{tick}°C</text>')
    labels = []
    for fraction in range(5):
        day = start + timedelta(days=round((end - start).days * fraction / 4))
        x, _ = point(day, ymin)
        labels.append(f'<text x="{x:.1f}" y="{height-pad+28}" text-anchor="middle">{day:%b %Y}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Daily temperature history and linear-regression forecast">
<style>text{{font-family:Arial,sans-serif;fill:#344054;font-size:14px}}.grid{{stroke:#d0d5dd;stroke-width:1}}.axis{{stroke:#667085;stroke-width:1.5}}.history{{fill:none;stroke:#2563eb;stroke-width:2}}.forecast{{fill:none;stroke:#f97316;stroke-width:2.5;stroke-dasharray:7 5}}</style>
<rect width="100%" height="100%" fill="white"/><text x="{pad}" y="32" font-size="20" font-weight="bold">Daily temperature and 30-day forecast</text>
{''.join(grid)}<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" class="axis"/>{''.join(labels)}
<path d="{path(rows)}" class="history"/><path d="{path([(rows[-1][0], rows[-1][1])] + forecasts)}" class="forecast"/>
<text x="{width-pad-190}" y="54" fill="#2563eb">— Observed</text><text x="{width-pad-190}" y="76" fill="#f97316">-- Forecast</text></svg>'''
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze daily temperature data and forecast future temperatures.")
    parser.add_argument("csv", nargs="?", type=Path, help="CSV file with date and temperature columns")
    parser.add_argument("--days", type=int, default=30, help="Forecast horizon in days (default: 30)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated files")
    parser.add_argument("--demo", action="store_true", help="Run with generated illustrative data")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be at least 1")
    if bool(args.csv) == args.demo:
        parser.error("provide a CSV path or use --demo")
    rows = demo_observations() if args.demo else load_observations(args.csv)
    slope, intercept, r_squared = linear_regression(rows)
    last_day = rows[-1][0]
    forecasts = [(last_day + timedelta(days=offset), intercept + slope * (last_day + timedelta(days=offset) - rows[0][0]).days) for offset in range(1, args.days + 1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "temperature_forecast.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "predicted_temperature_c"])
        writer.writerows((day.isoformat(), f"{temp:.2f}") for day, temp in forecasts)
    make_chart(rows, forecasts, args.output_dir / "temperature_trend.svg")
    observed = [temp for _, temp in rows]
    summary = f"""# Weather temperature analysis\n\n- Observations: {len(rows)} daily readings from {rows[0][0]:%Y-%m-%d} to {last_day:%Y-%m-%d}\n- Average observed temperature: {mean(observed):.2f} °C\n- Temperature variability (population standard deviation): {pstdev(observed):.2f} °C\n- Linear trend: {slope * 30:+.2f} °C per 30 days\n- Model fit (R²): {r_squared:.3f}\n- Forecast period: {(last_day + timedelta(days=1)):%Y-%m-%d} to {forecasts[-1][0]:%Y-%m-%d}\n- Predicted temperature on final forecast day: {forecasts[-1][1]:.2f} °C\n\nThis is a simple time-based linear regression. It is best used for broad direction, not daily weather certainty: seasonality, rainfall, and local effects are not modelled.\n"""
    (args.output_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
