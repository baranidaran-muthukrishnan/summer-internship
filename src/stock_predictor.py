"""Train a simple stock-price predictor from historical CSV data.

The model uses lagged closing prices, rolling averages, volatility, return,
and volume features. It fits a ridge regression model with the Python standard
library, so the project can run without installing NumPy, pandas, or sklearn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean


PRICE_COLUMNS = ("Close", "Adj Close", "close", "adj_close", "price")
DATE_COLUMNS = ("Date", "date", "timestamp")
VOLUME_COLUMNS = ("Volume", "volume")


@dataclass
class StockRow:
    date: datetime
    close: float
    volume: float


@dataclass
class Scaler:
    means: list[float]
    stds: list[float]

    def transform_row(self, row: list[float]) -> list[float]:
        return [
            (value - self.means[index]) / self.stds[index]
            for index, value in enumerate(row)
        ]

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        return [self.transform_row(row) for row in rows]


@dataclass
class RidgeModel:
    weights: list[float]
    scaler: Scaler
    feature_names: list[str]
    lookback: int

    def predict_one(self, features: list[float]) -> float:
        scaled = self.scaler.transform_row(features)
        return self.weights[0] + sum(
            weight * value for weight, value in zip(self.weights[1:], scaled)
        )


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value.replace(",", ""))


def pick_column(fieldnames: list[str], options: tuple[str, ...]) -> str:
    for option in options:
        if option in fieldnames:
            return option
    raise ValueError(f"Expected one of {options}; found columns {fieldnames}")


def load_stock_csv(path: Path) -> list[StockRow]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")

        date_column = pick_column(reader.fieldnames, DATE_COLUMNS)
        price_column = pick_column(reader.fieldnames, PRICE_COLUMNS)
        volume_column = (
            pick_column(reader.fieldnames, VOLUME_COLUMNS)
            if any(column in reader.fieldnames for column in VOLUME_COLUMNS)
            else None
        )

        rows = []
        for record in reader:
            date_text = record[date_column].strip()
            date = datetime.fromisoformat(date_text.replace("/", "-"))
            close = parse_float(record[price_column])
            volume = parse_float(record.get(volume_column), 0.0) if volume_column else 0.0
            rows.append(StockRow(date=date, close=close, volume=volume))

    rows.sort(key=lambda row: row.date)
    if len(rows) < 40:
        raise ValueError("Need at least 40 rows of history for train/test splitting.")
    return rows


def rolling(values: list[float], end: int, window: int) -> list[float]:
    start = max(0, end - window)
    return values[start:end]


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def make_feature_row(closes: list[float], volumes: list[float], index: int) -> list[float]:
    previous = closes[index - 1]
    lag_2 = closes[index - 2]
    lag_3 = closes[index - 3]
    lag_7 = closes[index - 7]
    ma_5 = mean(rolling(closes, index, 5))
    ma_10 = mean(rolling(closes, index, 10))
    ma_20 = mean(rolling(closes, index, 20))
    one_day_return = (previous - lag_2) / lag_2 if lag_2 else 0.0
    recent_returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(max(1, index - 10), index)
        if closes[i - 1]
    ]
    volatility_10 = stddev(recent_returns)
    recent_volume = rolling(volumes, index, 10)
    avg_volume = mean(recent_volume) if recent_volume else 0.0
    volume_ratio = volumes[index - 1] / avg_volume if avg_volume else 1.0
    return [
        previous,
        lag_2,
        lag_3,
        lag_7,
        ma_5,
        ma_10,
        ma_20,
        one_day_return,
        volatility_10,
        volume_ratio,
    ]


def build_features(rows: list[StockRow], lookback: int) -> tuple[list[list[float]], list[float], list[str]]:
    closes = [row.close for row in rows]
    volumes = [row.volume for row in rows]
    feature_names = [
        "lag_1_close",
        "lag_2_close",
        "lag_3_close",
        "lag_7_close",
        "ma_5",
        "ma_10",
        "ma_20",
        "return_1d",
        "volatility_10",
        "volume_ratio_10",
    ]
    features: list[list[float]] = []
    targets: list[float] = []

    for index in range(lookback, len(rows)):
        features.append(make_feature_row(closes, volumes, index))
        targets.append(closes[index])

    return features, targets, feature_names


def fit_scaler(rows: list[list[float]]) -> Scaler:
    columns = list(zip(*rows))
    means = [mean(column) for column in columns]
    stds = [stddev(list(column)) or 1.0 for column in columns]
    return Scaler(means=means, stds=stds)


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]

    for pivot in range(size):
        max_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        augmented[pivot], augmented[max_row] = augmented[max_row], augmented[pivot]
        if abs(augmented[pivot][pivot]) < 1e-12:
            raise ValueError("Regression matrix is singular; try a larger ridge value.")

        pivot_value = augmented[pivot][pivot]
        augmented[pivot] = [value / pivot_value for value in augmented[pivot]]

        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            augmented[row] = [
                current - factor * base
                for current, base in zip(augmented[row], augmented[pivot])
            ]

    return [augmented[row][-1] for row in range(size)]


def fit_ridge_regression(
    features: list[list[float]],
    targets: list[float],
    feature_names: list[str],
    lookback: int,
    alpha: float,
) -> RidgeModel:
    scaler = fit_scaler(features)
    scaled = scaler.transform(features)
    design = [[1.0] + row for row in scaled]
    width = len(design[0])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]

    for row, target in zip(design, targets):
        for i in range(width):
            xty[i] += row[i] * target
            for j in range(width):
                xtx[i][j] += row[i] * row[j]

    for i in range(1, width):
        xtx[i][i] += alpha

    return RidgeModel(
        weights=solve_linear_system(xtx, xty),
        scaler=scaler,
        feature_names=feature_names,
        lookback=lookback,
    )


def evaluate(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [prediction - truth for prediction, truth in zip(predicted, actual)]
    mae = mean(abs(error) for error in errors)
    rmse = math.sqrt(mean(error**2 for error in errors))
    mape = mean(
        abs(error / truth) for error, truth in zip(errors, actual) if truth != 0
    ) * 100
    baseline = actual[:-1]
    baseline_actual = actual[1:]
    baseline_mae = (
        mean(abs(previous - truth) for previous, truth in zip(baseline, baseline_actual))
        if baseline_actual
        else 0.0
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "baseline_naive_mae": baseline_mae,
    }


def split_train_test(
    features: list[list[float]],
    targets: list[float],
    test_size: float,
) -> tuple[list[list[float]], list[float], list[list[float]], list[float], int]:
    split_index = int(len(features) * (1 - test_size))
    split_index = min(max(split_index, 20), len(features) - 5)
    return (
        features[:split_index],
        targets[:split_index],
        features[split_index:],
        targets[split_index:],
        split_index,
    )


def append_synthetic_next_row(rows: list[StockRow], close: float) -> list[StockRow]:
    recent = rows[-10:]
    avg_volume = mean(row.volume for row in recent) if any(row.volume for row in recent) else 0.0
    next_date = rows[-1].date + timedelta(days=1)
    return rows + [StockRow(date=next_date, close=close, volume=avg_volume)]


def forecast_future(rows: list[StockRow], model: RidgeModel, days: int) -> list[dict[str, str | float]]:
    working_rows = rows[:]
    forecasts: list[dict[str, str | float]] = []
    for _ in range(days):
        closes = [row.close for row in working_rows]
        volumes = [row.volume for row in working_rows]
        next_features = make_feature_row(closes, volumes, len(working_rows))
        prediction = model.predict_one(next_features)
        next_date = working_rows[-1].date + timedelta(days=1)
        forecasts.append({"date": next_date.date().isoformat(), "predicted_close": prediction})
        working_rows = append_synthetic_next_row(working_rows, prediction)
    return forecasts


def save_predictions(
    path: Path,
    dates: list[datetime],
    actual: list[float],
    predicted: list[float],
    forecasts: list[dict[str, str | float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "date", "actual_close", "predicted_close"])
        for date, truth, prediction in zip(dates, actual, predicted):
            writer.writerow(["test", date.date().isoformat(), f"{truth:.4f}", f"{prediction:.4f}"])
        for forecast in forecasts:
            writer.writerow(
                [
                    "forecast",
                    forecast["date"],
                    "",
                    f"{float(forecast['predicted_close']):.4f}",
                ]
            )


def train_and_predict(args: argparse.Namespace) -> dict[str, object]:
    rows = load_stock_csv(Path(args.csv))
    features, targets, feature_names = build_features(rows, args.lookback)
    train_x, train_y, test_x, test_y, split_index = split_train_test(
        features, targets, args.test_size
    )
    model = fit_ridge_regression(
        train_x,
        train_y,
        feature_names=feature_names,
        lookback=args.lookback,
        alpha=args.alpha,
    )
    predictions = [model.predict_one(row) for row in test_x]
    metrics = evaluate(test_y, predictions)
    forecasts = forecast_future(rows, model, args.forecast_days)

    target_dates = [row.date for row in rows[args.lookback + split_index :]]
    save_predictions(Path(args.output), target_dates, test_y, predictions, forecasts)

    return {
        "rows_loaded": len(rows),
        "train_rows": len(train_x),
        "test_rows": len(test_x),
        "metrics": metrics,
        "forecast": forecasts,
        "prediction_file": args.output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict stock closing prices from CSV history.")
    parser.add_argument("--csv", required=True, help="Historical CSV with Date and Close columns.")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Where to save predictions.")
    parser.add_argument("--forecast-days", type=int, default=5, help="Number of future days to forecast.")
    parser.add_argument("--lookback", type=int, default=20, help="Minimum history used for features.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows used for testing.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_and_predict(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
