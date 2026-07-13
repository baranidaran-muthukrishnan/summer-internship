<<<<<<< HEAD
# Weather Data Analysis and Prediction

This project analyzes daily temperatures and makes a basic linear-regression forecast.

## Run the included demonstration

```powershell
python weather_analysis.py --demo
```

## Analyze your own data

```powershell
python weather_analysis.py path\to\weather.csv --days 30
```

The CSV needs a `date` column in `YYYY-MM-DD` format and one temperature column named `temperature`, `temp_c`, `temperature_c`, or `tavg`.

Generated files appear in `outputs/`:

- `analysis_summary.md` — key metrics and forecast interpretation
- `temperature_forecast.csv` — predicted daily temperatures
- `temperature_trend.svg` — observed history and the forecast line

The model is deliberately simple: it fits temperature against elapsed days. It shows long-run direction, but does not account for seasonality or other weather variables.
=======
# Stock Price Predictor

This project trains a small stock-price forecasting model from historical CSV data.
It uses lag features, moving averages, volatility, returns, volume ratios, and a
ridge regression model implemented with the Python standard library.

## Features

- Loads historical stock data from CSV.
- Accepts common column names such as `Date`, `Close`, `Adj Close`, and `Volume`.
- Builds time-series features from previous prices.
- Uses chronological train/test splitting to avoid look-ahead leakage.
- Reports MAE, RMSE, MAPE, and a naive baseline MAE.
- Writes test predictions and future forecasts to `outputs/predictions.csv`.

## Quick Start

Generate deterministic sample data:

```powershell
python src/generate_sample_data.py
```

Train and forecast:

```powershell
python src/stock_predictor.py --csv data/sample_stock_prices.csv --forecast-days 7
```

Use your own data:

```powershell
python src/stock_predictor.py --csv path/to/history.csv --output outputs/my_predictions.csv
```

The input CSV should contain at least:

```csv
Date,Close,Volume
2025-01-02,187.42,42100000
```

`Volume` is optional. If omitted, the model still runs with price-based features.

## Model Notes

This is a baseline predictive model, not investment advice. Stock prices are noisy,
non-stationary, and affected by news and market conditions that are not present in
historical price data alone. For production use, compare this baseline with stronger
methods such as random forests, gradient boosting, ARIMA/SARIMAX, LSTMs, or models
that include broader market and fundamental indicators.

## Output

The script prints a JSON summary and writes a CSV like:

```csv
type,date,actual_close,predicted_close
test,2024-10-16,171.25,170.88
forecast,2024-12-27,,183.12
```
>>>>>>> fa040087f306e54f0b780f1d8bdd7dfbd4c620a8
