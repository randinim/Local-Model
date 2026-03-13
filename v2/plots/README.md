# Plots README — Waste Predictor (v2, no capacity)

This README explains each plot generated in `v2/plots`, the use case for the plot, how to read it, and suggested follow-up actions. Use these when validating model behavior, identifying failure modes, and communicating results.

**Files produced**
- `per_target_r2.png`
- `parity_representative.png`
- `residuals_violin.png`
- `feature_importance_top15.png` (generated if GB importances available)
- `error_vs_production.png`
- `timeseries_total_waste.png`
- `metrics_summary.json`

---

**Per-target R² (per_target_r2.png)**
- Purpose: Summarize predictive quality per target (coarse ranking).
- What it shows: Bar chart of R² for each output target (sorted). Higher = better.
- How to read: Targets on the x-axis, R² on the y-axis. Values near 1 indicate excellent fit; values < 0.5 indicate poor predictive performance.
- Use cases: Identify weak targets (e.g., ion concentrations) for extra modeling effort or data collection. Prioritize features or alternate model strategies for low-R² targets.
- Actions: Select low-R² targets for separate diagnostics (SHAP, residual analysis, more features, or domain priors).

**Parity plots (parity_representative.png)**
- Purpose: Check calibration and bias for representative targets.
- What it shows: True vs predicted scatter with identity line for two representative outputs (Total_Waste_kg and a concentration).
- How to read: Points near the identity line indicate good calibration. Systematic offsets or slope deviations indicate bias or scale errors; spread indicates variance.
- Use cases: Validate whether predictions are unbiased and whether errors grow with magnitude.
- Actions: If biased, inspect target transforms (log/exp), retrain with corrected target scaling, or use calibration.

**Residual distributions (residuals_violin.png)**
- Purpose: Visualize error distributions across all targets compactly.
- What it shows: Violin plots of residuals (True − Predicted) for each target, ordered by R².
- How to read: Wide violins → heavy-tailed errors; asymmetric violins → bias (skew). Look for multi-modality (model missing regimes) or long tails (outliers).
- Use cases: Find targets with heteroscedasticity or skewed errors that may benefit from heteroscedastic models, per-bin models, or robust loss functions.
- Actions: For heavy tails, consider clipping, robust loss (Huber), or targeted data cleaning.

**Feature importance (feature_importance_top15.png)**
- Purpose: Show which input features the Gradient Boosting models used most (average across targets).
- What it shows: Mean feature importance for top features.
- How to read: Features at top are strong predictors (according to GB split gains). Note: importances reflect model usage, not causal effect.
- Use cases: Confirm domain expectations (e.g., evaporation proxies important), detect unexpected dominant features that may indicate leakage (e.g., features derived from targets).
- Actions: If an unwanted feature (like `production_capacity`) were dominant, remove/robustify it. Use important features for simpler interpretable models if needed.
- Note: This plot is only generated if the saved GB models expose `feature_importances_`.

**Error vs Production Volume (error_vs_production.png)**
- Purpose: Detect heteroscedasticity — whether error magnitude depends on production scale.
- What it shows: Scatter of residuals (True − Pred) vs production_volume with a smoothed trend.
- How to read: A flat trend → homoscedastic errors. A trend where error magnitude increases with production suggests heteroscedastic noise.
- Use cases: Decide whether a constant-variance loss is appropriate or if variance-scaling (e.g., log-transform, weighted loss) is needed.
- Actions: For heteroscedastic errors, consider target transforms, modeling variance explicitly, or stratified models by production band.

**Time-series true vs predicted (timeseries_total_waste.png)**
- Purpose: Inspect temporal alignment and drift across the dataset date range.
- What it shows: Chronological plot of true and predicted `Total_Waste_kg` over time.
- How to read: Close tracking over time indicates stable temporal modeling. Systematic divergence indicates temporal drift, seasonal mismatch, or out-of-sample regimes.
- Use cases: Detect model degradation over certain years/seasons and check seasonality capture.
- Actions: If drift appears, consider temporally-aware validation, retraining on recent data, or adding explicit time/seasonality features.

**Metrics summary (metrics_summary.json)**
- Purpose: Machine-readable summary of global metrics (R², MAE, RMSE) and per-target R².
- Use cases: Automated reporting, checkpointing, or comparison to other runs.
- Actions: Use for numeric comparisons during ablation studies or CI tests.

---

Recommended follow-ups (research workflow):
1. Run SHAP or permutation importance for the worst-performing targets to confirm drivers of poor performance.
2. Perform ablation experiments per-target (drop top features) to verify robustness and detect leakage.
3. If ion concentrations underperform, try specialized models for concentrations (physical priors or constrained output layers).
4. Add a `capacity_missing` indicator if you plan to impute capacity at inference, and retrain for robustness.

If you want, I can:
- Generate a SHAP report for the top 2 weak targets,
- Produce a side-by-side comparison README between the original model (with `production_capacity`) and this no-capacity run,
- Or add the README text into repository docs. Which do you prefer?
