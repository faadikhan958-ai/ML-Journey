# Lahore Air Quality (PM2.5 AQI) Prediction: Project 1

A self-sourced regression project: estimate Lahore's daily PM2.5 AQI from that day's weather, starting from a vague idea and a messy real dataset.

## Problem
Lahore regularly ranks among the world's most polluted cities. Can weather (temperature, humidity, wind, pressure) estimate the day's air-quality level, and which factors matter most?

> **Scope note:** the current model *estimates the same day's AQI* from that day's weather. Forecasting *tomorrow's* AQI needs lag features (for example yesterday's AQI); see next steps.

## Data
- Daily records from June 2019 to late 2023: 1,644 rows × 17 columns
- Target: `aqi_pm2.5`; features: weather columns (max/avg/min temperature, dew point, humidity, wind speed, pressure)
- Source: a public Kaggle dataset on Lahore air quality (daily AQI plus weather)

## Cleaning (real data problems I hit)
- **221 missing target values** (13%): rows dropped, never filled, because a target should not be fabricated. 1,423 rows remain.
- **Mixed date formats** (`29-08-23` next to `1/9/2023`): parsed with `dayfirst=True` and `errors='coerce'`, then verified that 0 dates failed to convert.
- **One value above the 500 AQI scale** (503 on 26 Dec 2021, smog season): capped at 500 instead of deleted, since it was a genuinely bad-air day.
- Cleaning steps wrapped in a reusable function (`clean_aqi_data`).

## Baseline
Always predicting the mean AQI (179.5) gives **MAE 63.05**, the number any model must beat.

## Features
The 15 weather columns, plus engineered features: `month`, `range_temp` (daily max minus min temperature), `range_humidity`, and `is_winter` (Nov to Feb, Lahore's smog season).

## Results
5-fold cross-validation with scaling inside a Pipeline (mean MAE ± std; lower is better)

| Model | Weather features only | + engineered features |
|---|---|---|
| Linear Regression | 43.86 ± 6.99 | 40.78 ± 7.80 |
| Random Forest | 39.39 ± 7.67 | **37.16 ± 6.19** |

Baseline: 63.05. Best model: Random Forest with engineered features, about **41% lower error than the baseline**, with the smallest spread across folds.

Earlier single 80/20 split: Random Forest MAE 34.72 without `month`, 32.34 with it.

**Time-aware check** (train on 1 Jun 2019 to 22 Jan 2023, test on 23 Jan to 30 Nov 2023; baseline uses the training mean only):

| Model | MAE |
|---|---|
| Baseline | 62.41 |
| Linear Regression | 31.51 |
| Random Forest | 27.77 |

Random Forest MAE in each of the five time-ordered CV folds: 42.7, 37.1, 33.6, 44.7, 27.6. The last fold is essentially the time-aware test above.

## What I learned
- **Model choice mattered more than feature engineering:** moving from Linear Regression to Random Forest cut about 4.5 MAE points, while the engineered features added 2 to 3. The weather-pollution relationship is non-linear.
- **Compare folds, not just means:** fold-to-fold std (6 to 8) was larger than the gain from the new features, so I compared base vs new fold by fold. The engineered features won in 4 of 5 folds: a modest but consistent gain, not a dramatic one.
- **Leakage, again:** a `0.0` MAE appeared when I left the target column in one feature set by mistake. A near-perfect score is a signal to audit the features.
- **Which days you test on matters a lot:** Random Forest MAE ranged from 27.6 to 44.7 across the five time-ordered folds. The same model scores 32.3 (random split), 27.8 (most recent 20% of days) or 37.2 (5-fold mean) depending on the test period, so I quote the cross-validated mean as the headline. I expected the time-ordered test to look *worse*; it looked better. That test window mostly misses the Dec-Jan smog peak, which probably explains why it was the easiest period. The ranking (Random Forest < Linear < baseline) held in every setup.

## Limitations and next steps
- Same-day estimation, not a true forecast. Next: lag features (previous days' AQI) to predict tomorrow.
- Some periods are much harder than others (the folds with errors of about 43 and 45). Next: find which dates and conditions drive those errors.
- Deeper error analysis of the largest errors.
- Deployment as a Streamlit app is planned.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook
```
Keep the CSV in the same folder as the notebook.
