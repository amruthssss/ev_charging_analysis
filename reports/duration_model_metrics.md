# Phase 5 Model 2 - Duration Prediction

- Run timestamp (UTC): 2025-12-14T09:20:34.828643Z
- Models compared: 2
- Best model: hist_gradient_boosting (MAE 51.69, RMSE 63.87, R2 0.020)
- Residual summary: D:\ev\data\processed\station_duration_residuals.csv

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| random_forest | 52.28 | 64.20 | 0.010 |
| hist_gradient_boosting | 51.69 | 63.87 | 0.020 |

## Best Model Hyperparameters
- model__l2_regularization: 0.5
- model__learning_rate: 0.03
- model__max_depth: 6
- model__max_leaf_nodes: 31
- model__min_samples_leaf: 20