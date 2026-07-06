# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-05

### Added
- Natural-language data analysis agent built on a LangChain tool-calling agent with Anthropic Claude, wrapped in a conversational Streamlit UI.
- Seven specialized tools: `explore_data` (EDA), `run_code` (Python execution on the DataFrame), `statistical_test`, `create_chart`, `train_model`, `analyze_time_series`, and `data_quality`.
- Statistical testing suite with 15+ tests (normality consensus, Welch's t-test, Mann-Whitney, ANOVA, Kruskal-Wallis, chi-square, Pearson/Spearman, VIF, ADF, KPSS, Durbin-Watson) including assumption checks and effect sizes.
- Interactive visualization engine with 14 Plotly chart types (histogram, bar, line, scatter, box, violin, heatmap, pie, area, treemap, scatter matrix, 2D density, and time series) using a dark theme.
- ML pipeline covering classification, regression, and clustering with baseline comparison, stratified k-fold cross-validation, and feature-importance charts (scikit-learn + XGBoost).
- Time series suite with ADF/KPSS stationarity testing, STL decomposition, ACF/PACF, rolling statistics, and ARIMA forecasting with confidence intervals.
- Data quality engine with MCAR/MAR/MNAR missing-value classification, IQR outlier detection, type validation, and distribution anomaly detection.
- CSV, Excel, and JSON dataset upload plus multi-turn conversation memory for iterative analysis.
- Project scaffolding: MIT license, `.env.example`, smoke tests, and a GitHub Actions CI workflow (Python 3.10-3.12) with Dependabot.

[0.1.0]: https://github.com/Kantamaniprakash/data-analysis-agent/releases/tag/v0.1.0
