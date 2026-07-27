# Time Series

Forecasting a series over time. Shared template: lag + rolling + calendar features, chronological split, naive and seasonal-naive baselines the model must beat, scored MAE / RMSE / MAPE / R2.

[Back to all projects](../README.md)

| Project | Description |
|---|---|
| [Weather Forecasting](Weather%20Forecasting) | Jena hourly temp, next-hour C, Linear MAE 0.39C / R2 0.995 (5x better than seasonal-naive). |
| [COVID-19 Trend Analysis](COVID-19%20Trend%20Analysis) | JHU global daily new cases, Linear R2 0.869, MAPE 16% (weekly reporting cycle). |
| [Website Traffic Forecasting](Website%20Traffic%20Forecasting) | Wikipedia 'Python' daily pageviews, RF R2 0.28 (noisy but beats baselines). |
| [Retail Sales Forecasting](Retail%20Sales%20Forecasting) | UCI Online Retail daily revenue, honest hard case (R2 -0.03), ML halves baseline error. |
| [Cryptocurrency Price Trend Analysis](Cryptocurrency%20Price%20Trend%20Analysis) | CoinGecko BTC daily, Ridge R2 0.968 (caveat: lag-1 persistence, not skill). |
| [Uber Ride Demand Forecasting](Uber%20Ride%20Demand%20Forecasting) | 538 NYC pickups -> 183 daily, Ridge R2 0.50, linear beats overfit trees. |

_6 projects in this category._
