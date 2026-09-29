# PHASE 0 — ARCHITECTURE, PROJECT SKELETON & IMPLEMENTATION PLAN
## AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System

---

## 1. ARCHITECTURE OVERVIEW

This system is a **production-oriented, multi-layered decision-support platform** for equity markets. It is not a price-prediction model — it is a complete intelligence pipeline from raw market data to portfolio decisions.

### Design Philosophy

| Principle | Expression |
|---|---|
| **Provider Independence** | All data sources accessed through abstract interfaces |
| **Model Independence** | All ML/DL models plug into a common prediction contract |
| **Point-in-Time Integrity** | Every historical query is bound by availability timestamps |
| **Explainability First** | Every prediction output carries its explanatory chain |
| **Risk-Gated Decisions** | Signals never bypass the risk and portfolio layers |
| **Reproducibility** | Every experiment, model, and backtest is fully versioned |
| **Separation of Concerns** | Data, features, models, risk, portfolio, decisions — each independently testable |
| **LLM as Explainer Only** | LLMs explain quant outputs; they do not produce numerical signals |

---

## 2. SYSTEM ARCHITECTURE DIAGRAM

```
╔══════════════════════════════════════════════════════════════╗
║                    EXTERNAL DATA SOURCES                     ║
║  Market Data │ Fundamentals │ Macro │ News │ Sentiment │ Alt ║
╚══════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════╗
║                    DATA INGESTION LAYER                      ║
║   Provider Adapters → Scheduler → Raw Storage → Audit Log   ║
╚══════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════╗
║                DATA VERIFICATION & QUALITY                   ║
║  Source Verify → Schema Check → Type/Range → Consistency    ║
║  Staleness → Outlier → Cross-Source Conflict → Quality State ║
╚══════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════╗
║              DATA NORMALIZATION & STORAGE                    ║
║    Security Master │ Market Calendar │ Point-in-Time Store   ║
╚══════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════╗
║                    FACTOR ENGINE                             ║
║  Fundamentals │ Technical │ Macro │ Sentiment │ Alternative  ║
╚══════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════╗
║                  FEATURE ENGINEERING                         ║
║  Raw → Transform → Roll → Lag → Normalize → Cross-Section   ║
║           Feature Store (versioned)                          ║
╚══════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════╗
║              MODEL / DATASET PIPELINE                        ║
║   Dataset Builder → Train/Val/Test Split → Experiment Track  ║
╚══════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════╗
║                  ML MODEL REGISTRY                           ║
║  Classical │ Tree │ RNN/LSTM/GRU │ CNN │ Attention │ NLP    ║
╚══════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════╗
║                  PREDICTION ENGINE                           ║
║  Expected Return │ Direction │ Price Range │ Uncertainty     ║
║  Prediction Strength │ Model Agreement │ Horizon            ║
╚══════════════════════════════════════════════════════════════╝
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
╔════════════════════╗         ╔════════════════════════╗
║  EXPLANATION LAYER ║         ║    RISK ENGINE         ║
║  SHAP │ Factors    ║         ║  Vol │ Beta │ VaR │     ║
║  Contributions     ║         ║  Drawdown │ Stress     ║
╚════════════════════╝         ╚════════════════════════╝
                                            │
                                            ▼
                             ╔══════════════════════════╗
                             ║    PORTFOLIO ENGINE      ║
                             ║  Allocation │ Optimize   ║
                             ║  Constraints │ Impact    ║
                             ╚══════════════════════════╝
                                            │
                                            ▼
                             ╔══════════════════════════╗
                             ║    DECISION ENGINE       ║
                             ║  BUY│HOLD│REDUCE│EXIT    ║
                             ║  Signal Lifecycle        ║
                             ╚══════════════════════════╝
                                            │
                    ┌───────────────────────┤
                    ▼                       ▼
       ╔═══════════════════╗   ╔═══════════════════════╗
       ║   BACKTESTING     ║   ║    PAPER TRADING      ║
       ║  Event-Driven     ║   ║  Simulated Execution  ║
       ║  Walk-Forward     ║   ║  P&L Tracking         ║
       ╚═══════════════════╝   ╚═══════════════════════╝
                                            │
                             ╔══════════════════════════╗
                             ║  REAL-TIME EVENT ENGINE  ║
                             ║  Event Bus │ Alerts      ║
                             ╚══════════════════════════╝
                                            │
                             ╔══════════════════════════╗
                             ║       API LAYER          ║
                             ║  FastAPI │ REST │ WS     ║
                             ╚══════════════════════════╝
                                            │
                             ╔══════════════════════════╗
                             ║      FRONTEND            ║
                             ║  Next.js │ TypeScript    ║
                             ╚══════════════════════════╝
```

---

## 3. EXACT REPOSITORY SKELETON

```
ai-stock-system/
│
├── backend/                          # Python / FastAPI application root
│   ├── app/
│   │   ├── __init__.py
│   │   │
│   │   ├── domain/                   # Pure domain — no framework imports
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── entities/             # Core business objects (pure Python dataclasses/Pydantic)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instrument.py     # Instrument, SecurityMaster
│   │   │   │   ├── company.py        # Company entity
│   │   │   │   ├── exchange.py       # Exchange entity
│   │   │   │   ├── price.py          # OHLCV bar, tick
│   │   │   │   ├── corporate_action.py
│   │   │   │   ├── fundamental.py    # Fundamental data entity
│   │   │   │   ├── macro.py          # Macro/economic data entity
│   │   │   │   ├── news_event.py     # News article, event entity
│   │   │   │   ├── sentiment.py      # Sentiment reading entity
│   │   │   │   ├── feature.py        # Feature, FeatureVersion
│   │   │   │   ├── prediction.py     # Prediction output entity
│   │   │   │   ├── signal.py         # Signal entity with lifecycle states
│   │   │   │   ├── portfolio.py      # Portfolio, Holding entity
│   │   │   │   ├── order.py          # Order entity
│   │   │   │   ├── fill.py           # Fill entity
│   │   │   │   ├── risk_metric.py    # RiskMetric entity
│   │   │   │   ├── model_version.py  # ML model version entity
│   │   │   │   ├── experiment.py     # Experiment entity
│   │   │   │   ├── backtest.py       # Backtest run entity
│   │   │   │   ├── alert.py          # Alert definition entity
│   │   │   │   └── data_quality.py   # DataQualityRecord entity
│   │   │   │
│   │   │   ├── enums/                # System-wide enumerations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── quality_state.py  # VALID, WARNING, STALE, CONFLICTED, QUARANTINED, UNAVAILABLE
│   │   │   │   ├── signal_state.py   # NEW, ACTIVE, STRENGTHENED, WEAKENED, INVALIDATED, CLOSED
│   │   │   │   ├── decision_type.py  # BUY, HOLD, REDUCE, EXIT, WAIT
│   │   │   │   ├── model_stage.py    # DEVELOPMENT, VALIDATION, CANDIDATE, PRODUCTION, RETIRED
│   │   │   │   ├── prediction_strength.py  # VERY_STRONG, STRONG, MODERATE, DEVELOPING, LIMITED
│   │   │   │   ├── horizon.py        # INTRADAY, DAY_1, DAY_5, MONTH_1, etc.
│   │   │   │   ├── asset_class.py    # EQUITY, ETF, INDEX, etc.
│   │   │   │   ├── market_regime.py  # TRENDING, HIGH_VOL, LOW_VOL, RISK_ON, RISK_OFF, etc.
│   │   │   │   └── data_source_type.py
│   │   │   │
│   │   │   ├── interfaces/           # Abstract contracts (ABC / Protocol) — no implementation
│   │   │   │   ├── __init__.py
│   │   │   │   ├── market_data_provider.py     # IMarketDataProvider
│   │   │   │   ├── fundamental_data_provider.py # IFundamentalDataProvider
│   │   │   │   ├── macro_data_provider.py      # IMacroDataProvider
│   │   │   │   ├── news_provider.py            # INewsProvider
│   │   │   │   ├── sentiment_provider.py       # ISentimentProvider
│   │   │   │   ├── alternative_data_provider.py # IAlternativeDataProvider
│   │   │   │   ├── feature_engine.py           # IFeatureEngine
│   │   │   │   ├── factor_calculator.py        # IFactorCalculator
│   │   │   │   ├── prediction_engine.py        # IPredictionEngine
│   │   │   │   ├── risk_engine.py              # IRiskEngine
│   │   │   │   ├── portfolio_engine.py         # IPortfolioEngine
│   │   │   │   ├── decision_engine.py          # IDecisionEngine
│   │   │   │   ├── backtest_engine.py          # IBacktestEngine
│   │   │   │   ├── event_bus.py                # IEventBus
│   │   │   │   ├── model_registry.py           # IModelRegistry
│   │   │   │   ├── experiment_tracker.py       # IExperimentTracker
│   │   │   │   ├── feature_store.py            # IFeatureStore
│   │   │   │   ├── market_calendar.py          # IMarketCalendar
│   │   │   │   ├── security_master.py          # ISecurityMaster
│   │   │   │   ├── data_verifier.py            # IDataVerifier
│   │   │   │   ├── data_quality_engine.py      # IDataQualityEngine
│   │   │   │   ├── explainer.py                # IExplainer
│   │   │   │   ├── alert_engine.py             # IAlertEngine
│   │   │   │   └── paper_trading_engine.py     # IPaperTradingEngine
│   │   │   │
│   │   │   └── value_objects/        # Immutable domain value types
│   │   │       ├── __init__.py
│   │   │       ├── money.py          # Money(amount, currency)
│   │   │       ├── price_range.py    # PriceRange(low, high, confidence)
│   │   │       ├── time_range.py     # TimeRange(start, end, tz)
│   │   │       ├── pit_timestamp.py  # PointInTimeTimestamp(event_time, publication_time, availability_time, ingestion_time)
│   │   │       ├── factor_score.py   # FactorScore(factor_id, value, z_score, regime)
│   │   │       └── prediction_interval.py  # PredictionInterval(lower, upper, confidence_level)
│   │   │
│   │   ├── infrastructure/           # All external concerns (DB, HTTP, queues)
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── database/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── session.py        # SQLAlchemy session factory, engine setup
│   │   │   │   ├── base.py           # SQLAlchemy declarative base
│   │   │   │   └── models/           # SQLAlchemy ORM models (one file per domain entity group)
│   │   │   │       ├── __init__.py
│   │   │   │       ├── instrument_models.py
│   │   │   │       ├── price_models.py
│   │   │   │       ├── corporate_action_models.py
│   │   │   │       ├── fundamental_models.py
│   │   │   │       ├── macro_models.py
│   │   │   │       ├── news_models.py
│   │   │   │       ├── feature_models.py
│   │   │   │       ├── prediction_models.py
│   │   │   │       ├── signal_models.py
│   │   │   │       ├── portfolio_models.py
│   │   │   │       ├── risk_models.py
│   │   │   │       ├── model_registry_models.py
│   │   │   │       ├── experiment_models.py
│   │   │   │       ├── backtest_models.py
│   │   │   │       ├── paper_trading_models.py
│   │   │   │       ├── alert_models.py
│   │   │   │       └── audit_models.py
│   │   │   │
│   │   │   ├── providers/            # Concrete data provider adapters
│   │   │   │   ├── __init__.py
│   │   │   │   ├── dev/              # Development-only providers (e.g. yfinance wrapper)
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── yfinance_market_data.py
│   │   │   │   │   └── yfinance_fundamental_data.py
│   │   │   │   └── production/       # Production provider adapters (implemented later)
│   │   │   │       └── __init__.py
│   │   │   │
│   │   │   ├── repositories/         # Data access objects (DB ↔ domain)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instrument_repository.py
│   │   │   │   ├── price_repository.py
│   │   │   │   ├── corporate_action_repository.py
│   │   │   │   ├── fundamental_repository.py
│   │   │   │   ├── macro_repository.py
│   │   │   │   ├── news_repository.py
│   │   │   │   ├── feature_repository.py
│   │   │   │   ├── prediction_repository.py
│   │   │   │   ├── signal_repository.py
│   │   │   │   ├── portfolio_repository.py
│   │   │   │   ├── risk_repository.py
│   │   │   │   ├── model_registry_repository.py
│   │   │   │   ├── experiment_repository.py
│   │   │   │   ├── backtest_repository.py
│   │   │   │   ├── paper_trading_repository.py
│   │   │   │   ├── alert_repository.py
│   │   │   │   └── audit_repository.py
│   │   │   │
│   │   │   ├── cache/                # Redis or in-memory cache layer
│   │   │   │   ├── __init__.py
│   │   │   │   └── cache_client.py
│   │   │   │
│   │   │   ├── messaging/            # Async event/message transport
│   │   │   │   ├── __init__.py
│   │   │   │   ├── in_process_event_bus.py   # Simple sync bus for dev
│   │   │   │   └── redis_event_bus.py        # Production async bus
│   │   │   │
│   │   │   └── storage/              # File/blob storage (model artifacts, feature snapshots)
│   │   │       ├── __init__.py
│   │   │       └── local_artifact_store.py
│   │   │
│   │   ├── application/              # Use-case orchestrators (no business logic)
│   │   │   ├── __init__.py
│   │   │   ├── ingestion/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ingest_prices_use_case.py
│   │   │   │   ├── ingest_fundamentals_use_case.py
│   │   │   │   ├── ingest_macro_use_case.py
│   │   │   │   └── ingest_news_use_case.py
│   │   │   ├── features/
│   │   │   │   ├── __init__.py
│   │   │   │   └── compute_features_use_case.py
│   │   │   ├── predictions/
│   │   │   │   ├── __init__.py
│   │   │   │   └── run_prediction_use_case.py
│   │   │   ├── portfolio/
│   │   │   │   ├── __init__.py
│   │   │   │   └── build_portfolio_use_case.py
│   │   │   ├── decisions/
│   │   │   │   ├── __init__.py
│   │   │   │   └── generate_decision_use_case.py
│   │   │   ├── backtesting/
│   │   │   │   ├── __init__.py
│   │   │   │   └── run_backtest_use_case.py
│   │   │   ├── paper_trading/
│   │   │   │   ├── __init__.py
│   │   │   │   └── execute_paper_trade_use_case.py
│   │   │   └── alerts/
│   │   │       ├── __init__.py
│   │   │       └── evaluate_alerts_use_case.py
│   │   │
│   │   ├── services/                 # Domain service implementations
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── security_master/
│   │   │   │   ├── __init__.py
│   │   │   │   └── security_master_service.py
│   │   │   │
│   │   │   ├── market_calendar/
│   │   │   │   ├── __init__.py
│   │   │   │   └── market_calendar_service.py
│   │   │   │
│   │   │   ├── data_verification/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── source_verifier.py
│   │   │   │   ├── schema_validator.py
│   │   │   │   └── provenance_tracker.py
│   │   │   │
│   │   │   ├── data_quality/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── quality_engine.py
│   │   │   │   ├── checks/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── schema_check.py
│   │   │   │   │   ├── type_check.py
│   │   │   │   │   ├── missing_values_check.py
│   │   │   │   │   ├── duplicate_check.py
│   │   │   │   │   ├── timestamp_check.py
│   │   │   │   │   ├── range_check.py
│   │   │   │   │   ├── consistency_check.py
│   │   │   │   │   ├── staleness_check.py
│   │   │   │   │   ├── outlier_check.py
│   │   │   │   │   └── cross_source_conflict_check.py
│   │   │   │   └── quality_state_resolver.py
│   │   │   │
│   │   │   ├── factors/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── fundamental_factors.py    # Revenue, EPS, margins, ROE, ROIC, debt, etc.
│   │   │   │   ├── technical_factors.py      # Returns, MAs, RSI, MACD, ATR, momentum, etc.
│   │   │   │   ├── macro_factors.py          # Index, rates, FX, commodities, breadth, etc.
│   │   │   │   ├── sentiment_factors.py      # News sentiment, event sentiment, unusual activity
│   │   │   │   └── alternative_factors.py    # Alt data factors (to be defined per source)
│   │   │   │
│   │   │   ├── feature_engineering/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transformers.py           # Log, diff, ratio, normalization transforms
│   │   │   │   ├── rolling_features.py       # Rolling window computations
│   │   │   │   ├── lagged_features.py        # Lag operators
│   │   │   │   ├── cross_sectional_features.py # Rank, z-score across universe
│   │   │   │   ├── regime_features.py        # Regime-conditioned features
│   │   │   │   ├── interaction_features.py   # Cross-factor interactions
│   │   │   │   └── feature_store_service.py  # Feature versioning and retrieval
│   │   │   │
│   │   │   ├── regime/
│   │   │   │   ├── __init__.py
│   │   │   │   └── regime_detector.py        # Market regime classification
│   │   │   │
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── registry/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── model_registry_service.py
│   │   │   │   ├── training/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── dataset_builder.py    # PIT-safe train/val/test dataset construction
│   │   │   │   │   ├── splitter.py           # Walk-forward, expanding window, rolling window
│   │   │   │   │   └── trainer.py            # Model training orchestrator
│   │   │   │   ├── classical/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── linear_models.py      # Ridge, Lasso, ElasticNet
│   │   │   │   │   ├── tree_models.py        # RandomForest, XGBoost, LightGBM
│   │   │   │   │   └── base_classical.py
│   │   │   │   ├── deep_learning/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── rnn_models.py         # RNN, LSTM, GRU
│   │   │   │   │   ├── cnn_models.py         # 1D CNN for time-series
│   │   │   │   │   ├── attention_models.py   # Attention, Transformer for time-series
│   │   │   │   │   └── base_deep.py          # PyTorch base module
│   │   │   │   ├── nlp/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── finbert_embedder.py   # FinBERT sentence embeddings
│   │   │   │   │   ├── sentiment_scorer.py   # News/event sentiment scoring
│   │   │   │   │   └── information_extractor.py # Structured info from text
│   │   │   │   └── ensemble/
│   │   │   │       ├── __init__.py
│   │   │   │       └── ensemble_combiner.py  # Model blending/stacking
│   │   │   │
│   │   │   ├── prediction/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── prediction_engine.py      # Orchestrates model ensemble → PredictionOutput
│   │   │   │   ├── uncertainty_estimator.py  # Conformal prediction, MC-Dropout, etc.
│   │   │   │   ├── calibrator.py             # Probability calibration
│   │   │   │   └── prediction_strength_calculator.py
│   │   │   │
│   │   │   ├── explainability/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── shap_explainer.py         # SHAP values computation
│   │   │   │   ├── feature_importance.py     # Model-native feature importance
│   │   │   │   ├── factor_contribution.py    # Factor-level contribution attribution
│   │   │   │   └── llm_narrator.py           # LLM explanation of quant outputs
│   │   │   │
│   │   │   ├── risk/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── risk_engine.py            # Top-level risk orchestrator
│   │   │   │   ├── volatility_calculator.py
│   │   │   │   ├── beta_calculator.py
│   │   │   │   ├── correlation_calculator.py
│   │   │   │   ├── var_calculator.py         # Value at Risk, CVaR
│   │   │   │   ├── drawdown_analyzer.py
│   │   │   │   ├── liquidity_assessor.py
│   │   │   │   ├── stress_tester.py
│   │   │   │   └── factor_exposure_calculator.py
│   │   │   │
│   │   │   ├── portfolio/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── portfolio_engine.py       # Main portfolio orchestrator
│   │   │   │   ├── optimizers/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── equal_weight.py
│   │   │   │   │   ├── min_variance.py
│   │   │   │   │   ├── max_sharpe.py
│   │   │   │   │   ├── risk_parity.py
│   │   │   │   │   ├── hierarchical_risk_parity.py
│   │   │   │   │   └── black_litterman.py
│   │   │   │   ├── position_sizer.py
│   │   │   │   └── constraints_checker.py
│   │   │   │
│   │   │   ├── decision/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decision_engine.py        # Multi-factor decision logic
│   │   │   │   ├── signal_manager.py         # Signal lifecycle management
│   │   │   │   └── thesis_invalidator.py     # Detects when signal thesis breaks
│   │   │   │
│   │   │   ├── backtesting/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── backtest_engine.py        # Event-driven backtester core
│   │   │   │   ├── event_queue.py
│   │   │   │   ├── order_executor.py         # Simulated fill with slippage/costs
│   │   │   │   ├── performance_calculator.py # Sharpe, Sortino, drawdown, etc.
│   │   │   │   └── bias_guard.py             # Look-ahead, survivorship, leakage checks
│   │   │   │
│   │   │   ├── paper_trading/
│   │   │   │   ├── __init__.py
│   │   │   │   └── paper_trading_engine.py
│   │   │   │
│   │   │   ├── realtime/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── event_processor.py        # Market event → state update → inference trigger
│   │   │   │   ├── inference_gate.py         # Decides whether to run inference
│   │   │   │   └── state_manager.py          # In-memory market state
│   │   │   │
│   │   │   ├── alerts/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── alert_engine.py
│   │   │   │   └── alert_evaluators/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── price_alert_evaluator.py
│   │   │   │       ├── prediction_alert_evaluator.py
│   │   │   │       ├── risk_alert_evaluator.py
│   │   │   │       ├── news_alert_evaluator.py
│   │   │   │       └── data_quality_alert_evaluator.py
│   │   │   │
│   │   │   └── monitoring/
│   │   │       ├── __init__.py
│   │   │       ├── model_monitor.py          # Prediction accuracy, calibration drift
│   │   │       ├── data_drift_monitor.py     # Feature distribution drift
│   │   │       └── system_health_monitor.py  # Data freshness, latency, provider status
│   │   │
│   │   ├── api/                       # FastAPI layer
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # FastAPI app factory
│   │   │   ├── dependencies.py        # Dependency injection (DB session, services)
│   │   │   ├── middleware/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth_middleware.py
│   │   │   │   ├── rate_limit_middleware.py
│   │   │   │   └── audit_middleware.py
│   │   │   ├── auth/
│   │   │   │   ├── __init__.py
│   │   │   │   └── jwt_handler.py
│   │   │   ├── schemas/               # Pydantic request/response schemas (not DB models)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── market_schemas.py
│   │   │   │   ├── fundamental_schemas.py
│   │   │   │   ├── feature_schemas.py
│   │   │   │   ├── prediction_schemas.py
│   │   │   │   ├── portfolio_schemas.py
│   │   │   │   ├── risk_schemas.py
│   │   │   │   ├── signal_schemas.py
│   │   │   │   ├── backtest_schemas.py
│   │   │   │   ├── paper_trading_schemas.py
│   │   │   │   └── alert_schemas.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── market.py          # /api/v1/market
│   │   │       ├── stocks.py          # /api/v1/stocks
│   │   │       ├── fundamentals.py    # /api/v1/fundamentals
│   │   │       ├── features.py        # /api/v1/features
│   │   │       ├── predictions.py     # /api/v1/predictions
│   │   │       ├── research.py        # /api/v1/research
│   │   │       ├── portfolio.py       # /api/v1/portfolio
│   │   │       ├── risk.py            # /api/v1/risk
│   │   │       ├── signals.py         # /api/v1/signals
│   │   │       ├── backtests.py       # /api/v1/backtests
│   │   │       ├── paper_trading.py   # /api/v1/paper-trading
│   │   │       └── alerts.py          # /api/v1/alerts
│   │   │
│   │   └── core/                      # Cross-cutting: config, logging, exceptions
│   │       ├── __init__.py
│   │       ├── config.py              # Pydantic Settings — loads from env
│   │       ├── logging.py             # Structured logging setup
│   │       ├── exceptions.py          # Domain exception hierarchy
│   │       └── constants.py           # System-wide constants
│   │
│   ├── alembic/                       # Database migrations
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/                  # Migration scripts (auto-generated)
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                # Shared fixtures
│   │   ├── unit/
│   │   │   ├── domain/
│   │   │   │   ├── test_entities.py
│   │   │   │   └── test_value_objects.py
│   │   │   ├── services/
│   │   │   │   ├── test_data_quality.py
│   │   │   │   ├── test_factor_calculators.py
│   │   │   │   ├── test_feature_engineering.py
│   │   │   │   ├── test_risk_engine.py
│   │   │   │   ├── test_portfolio_engine.py
│   │   │   │   ├── test_decision_engine.py
│   │   │   │   └── test_signal_manager.py
│   │   │   └── models/
│   │   │       ├── test_classical_models.py
│   │   │       └── test_deep_models.py
│   │   ├── integration/
│   │   │   ├── test_data_pipeline.py  # End-to-end ingestion → DB
│   │   │   ├── test_feature_pipeline.py
│   │   │   ├── test_prediction_pipeline.py
│   │   │   └── test_api_routes.py
│   │   ├── financial/                 # Financial correctness tests (separate from SW correctness)
│   │   │   ├── test_pit_integrity.py  # Verify no look-ahead bias
│   │   │   ├── test_backtesting_bias.py
│   │   │   ├── test_factor_math.py    # Verify factor formulas
│   │   │   └── test_risk_math.py
│   │   ├── contracts/                 # Provider contract tests
│   │   │   ├── test_market_data_provider_contract.py
│   │   │   ├── test_fundamental_provider_contract.py
│   │   │   └── test_macro_provider_contract.py
│   │   └── e2e/
│   │       └── test_full_pipeline.py
│   │
│   ├── scripts/                       # Operational scripts (run manually, not imported)
│   │   ├── backfill_prices.py
│   │   ├── load_security_master.py
│   │   ├── run_backtest.py
│   │   ├── evaluate_model.py
│   │   └── check_data_quality.py
│   │
│   ├── Dockerfile                     # Backend container image
│   ├── pyproject.toml                 # Python project metadata + dependencies
│   ├── requirements.txt               # Pinned production requirements
│   ├── requirements-dev.txt           # Dev/test requirements
│   └── alembic.ini                    # Alembic configuration
│
├── frontend/                          # Next.js application (implemented separately)
│   ├── app/
│   │   ├── layout.js
│   │   ├── page.js
│   │   └── components/
│   ├── public/
│   ├── package.json
│   └── next.config.js
│
├── docker/                            # Docker orchestration
│   ├── docker-compose.yml             # Backend + PostgreSQL + Redis (dev)
│   ├── docker-compose.prod.yml        # Production overrides
│   └── nginx/
│       └── nginx.conf                 # Reverse proxy (production)
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Lint, test, type-check on PRs
│       └── cd.yml                     # Deploy on merge to main
│
├── docs/
│   ├── architecture.md                # This document (maintained)
│   ├── data_providers.md              # Provider evaluation notes
│   ├── factor_definitions.md          # Each factor's formula/source
│   ├── model_experiments.md           # Experiment log
│   └── api_reference.md              # API contract documentation
│
├── .gitignore
├── .env.example                       # Env variable template (never commit .env)
└── README.md
```

---

## 4. FOLDER/FILE RESPONSIBILITY TABLE

| Path | Exists For | Does NOT Contain | Phase |
|---|---|---|---|
| `domain/entities/` | Pure business objects; no DB, no framework imports | ORM models, HTTP schemas | Phase 1 |
| `domain/interfaces/` | Abstract contracts (ABCs/Protocols) only | Implementation code | Phase 1 |
| `domain/enums/` | System-wide named states/types | Business logic | Phase 1 |
| `domain/value_objects/` | Immutable types with equality by value | Mutable state | Phase 1 |
| `infrastructure/database/models/` | SQLAlchemy ORM mappings | Domain logic | Phase 2 |
| `infrastructure/providers/dev/` | yfinance and other dev-only adapters | Production credentials | Phase 3 |
| `infrastructure/providers/production/` | Real market data provider adapters | Dev stubs | Phase 8+ |
| `infrastructure/repositories/` | DB query isolation | Business rules | Phase 2 |
| `infrastructure/messaging/` | Event transport | Business events definitions | Phase 10 |
| `services/data_verification/` | Source/schema/provenance checks | ML logic | Phase 3 |
| `services/data_quality/checks/` | Individual quality check rules | Factor computation | Phase 3 |
| `services/factors/` | Factor formula implementations | Raw data access | Phase 4 |
| `services/feature_engineering/` | Transform, roll, lag, normalize, cross-section | Model training | Phase 4 |
| `services/models/classical/` | sklearn, XGBoost, LightGBM wrappers | PyTorch | Phase 5 |
| `services/models/deep_learning/` | PyTorch RNN/LSTM/GRU/CNN/Attention | Classical ML | Phase 5 |
| `services/models/nlp/` | FinBERT, HuggingFace models | Numerical predictions | Phase 5 |
| `services/prediction/` | Ensemble orchestration, uncertainty, strength | Individual model training | Phase 6 |
| `services/explainability/` | SHAP, feature importance, LLM narration | Prediction generation | Phase 6 |
| `services/risk/` | Quantitative risk calculations | Portfolio allocation | Phase 7 |
| `services/portfolio/` | Optimization, sizing, constraints | Decisions | Phase 7 |
| `services/decision/` | Multi-factor decision + signal lifecycle | Portfolio math | Phase 8 |
| `services/backtesting/` | Event-driven simulation | Live execution | Phase 9 |
| `services/paper_trading/` | Paper position management | Real orders | Phase 10 |
| `services/realtime/` | Market event processing, inference gating | Heavy ML training | Phase 11 |
| `services/alerts/` | Alert rule evaluation | Alert delivery | Phase 11 |
| `services/monitoring/` | Drift, accuracy, health tracking | Business alerts | Phase 12 |
| `application/` | Thin use-case orchestrators | Business logic | Phase 2+ |
| `api/routes/` | HTTP endpoints, request/response mapping | Business logic | Phase 2+ |
| `api/schemas/` | Pydantic request/response models | DB models | Phase 2+ |
| `core/` | Config, logging, exceptions, constants | Feature logic | Phase 1 |
| `alembic/versions/` | DB migration scripts | Application code | Phase 2 |
| `tests/financial/` | Financial-correctness assertions | Software unit tests | Phase 3+ |
| `tests/contracts/` | Provider interface compliance | Implementation tests | Phase 3 |
| `docker/` | Container orchestration files | Application code | Phase 1 |
| `.github/workflows/` | CI/CD pipeline definitions | Application code | Phase 1 |
| `docs/` | Living architecture and experiment documentation | Code | All phases |
| `scripts/` | One-off operational CLI scripts | Importable modules | Phase 3+ |

---

## 5. DOMAIN BOUNDARIES

```
┌─────────────────────────────────────────────────────────────┐
│                        DOMAIN LAYER                         │
│  entities/ │ interfaces/ │ enums/ │ value_objects/          │
│  ─ Pure Python                                              │
│  ─ No framework imports                                     │
│  ─ No database imports                                      │
│  ─ No HTTP imports                                          │
│  ─ Fully testable in isolation                              │
└─────────────────────────────────────────────────────────────┘
         ↑ depends on nothing external
         
┌─────────────────────────────────────────────────────────────┐
│                      SERVICES LAYER                         │
│  Implements domain interfaces                               │
│  ─ Business logic lives here                                │
│  ─ Calls domain entities / value objects                    │
│  ─ Calls infrastructure via interfaces (never directly)     │
└─────────────────────────────────────────────────────────────┘
         ↑ depends on domain

┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│  Use-case orchestrators                                     │
│  ─ No business logic                                        │
│  ─ Calls services                                           │
│  ─ Translates between API schemas and domain entities       │
└─────────────────────────────────────────────────────────────┘
         ↑ depends on services + domain

┌─────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                       │
│  Database, HTTP clients, queues, file storage               │
│  ─ Implements interfaces defined in domain                  │
│  ─ Framework-specific code (SQLAlchemy, httpx, etc.)        │
│  ─ Never imported by domain layer                           │
└─────────────────────────────────────────────────────────────┘
         ↑ depends on domain interfaces

┌─────────────────────────────────────────────────────────────┐
│                      API LAYER                              │
│  FastAPI routers, schemas, middleware                        │
│  ─ Calls application layer use-cases only                   │
│  ─ Translates HTTP ↔ domain                                 │
└─────────────────────────────────────────────────────────────┘
         ↑ depends on application layer
```

**Strict rule:** Inner layers NEVER import outer layers.

---

## 6. DEPENDENCY FLOW

```
domain/ (no deps)
    ↑
infrastructure/ (implements domain interfaces)
    ↑
services/ (implements business logic using domain + infrastructure interfaces)
    ↑
application/ (orchestrates services)
    ↑
api/ (exposes application via HTTP)

Cross-cutting (injected everywhere):
  core/config.py → loaded at startup
  core/logging.py → used by all layers
  core/exceptions.py → raised by services, caught by API
```

**Data dependency flow within services:**

```
providers → data_verification → data_quality → factors → feature_engineering
    → dataset_builder → model_training → prediction_engine → explainability
    → risk_engine → portfolio_engine → decision_engine → signal_manager
    → backtesting / paper_trading / realtime → alerts → monitoring
```

---

## 7. DATABASE ENTITY MAP

> These are **conceptual entities** only. SQLAlchemy models will be created in Phase 2.

### Reference / Master Data

| Entity | Key Attributes |
|---|---|
| `Exchange` | id, code, name, country, timezone, market_open, market_close |
| `Instrument` | id, symbol, isin, exchange_id, company_id, asset_class, listing_status, trading_status |
| `Company` | id, name, cik, sector, industry, country |
| `InstrumentHistory` | instrument_id, symbol, name, effective_from, effective_to, change_reason |
| `MarketHoliday` | exchange_id, date, description |

### Time-Series Data

| Entity | Key Attributes |
|---|---|
| `PriceBar` | instrument_id, timestamp, open, high, low, close, volume, adjusted_close, provider_id, quality_state, pit_available_at |
| `CorporateAction` | instrument_id, action_type, ex_date, record_date, payment_date, ratio, amount, currency, provider_id |
| `FundamentalRecord` | instrument_id, period_end, fiscal_year, fiscal_quarter, metric_name, value, currency, source_id, pit_published_at, pit_available_at |
| `MacroDataPoint` | series_id, timestamp, value, unit, provider_id, pit_available_at |
| `NewsEvent` | id, headline, body, source, published_at, retrieved_at, available_at, instruments[], claim_type (FACT/CLAIM/OPINION/RUMOR) |

### Data Quality & Provenance

| Entity | Key Attributes |
|---|---|
| `DataSource` | id, name, provider_type, base_url, api_version, tier |
| `DataQualityRecord` | data_entity_type, data_entity_id, check_name, state, detail, checked_at |
| `ProvenanceRecord` | entity_type, entity_id, source_id, event_time, publication_time, retrieval_time, availability_time |

### Feature & Model

| Entity | Key Attributes |
|---|---|
| `FeatureDefinition` | id, name, group (FUNDAMENTAL/TECHNICAL/MACRO/SENTIMENT/ALT), formula_version |
| `FeatureValue` | feature_id, instrument_id, as_of_date, value, quality_state, computed_at |
| `FeatureVersion` | feature_id, version, effective_from, definition_hash |
| `Dataset` | id, name, feature_version, instruments[], date_range, split_config, created_at |
| `ModelVersion` | id, model_family, model_name, stage, training_period, validation_period, test_period, hyperparameters, seed, metrics, git_commit, artifact_path |
| `Experiment` | id, name, dataset_id, model_version_id, feature_version, hypothesis, results, created_at |

### Prediction & Signal

| Entity | Key Attributes |
|---|---|
| `Prediction` | id, instrument_id, model_version_id, horizon, expected_return, direction, price_lower, price_upper, prediction_strength, uncertainty, model_agreement, data_quality_state, feature_version, predicted_at |
| `Explanation` | prediction_id, shap_values (JSON), top_factors (JSON), model_contributions (JSON), generated_at |
| `Signal` | id, instrument_id, state, decision_type, prediction_id, reasons (JSON), factors (JSON), invalidation_condition, created_at, closed_at, closing_reason |

### Portfolio & Execution

| Entity | Key Attributes |
|---|---|
| `Portfolio` | id, name, description, currency, strategy, created_at |
| `Holding` | portfolio_id, instrument_id, quantity, average_cost, current_value, as_of_date |
| `Order` | id, portfolio_id, instrument_id, order_type, direction, quantity, limit_price, status, created_at |
| `Fill` | order_id, filled_quantity, fill_price, commission, filled_at |
| `CashBalance` | portfolio_id, currency, amount, as_of_date |

### Risk

| Entity | Key Attributes |
|---|---|
| `RiskMetric` | portfolio_id (or instrument_id), metric_name, value, computation_date, methodology |

### Backtesting & Paper Trading

| Entity | Key Attributes |
|---|---|
| `BacktestRun` | id, name, strategy_config, universe, start_date, end_date, model_version_id, feature_version, slippage_model, cost_model, results (JSON), created_at |
| `BacktestTrade` | backtest_id, instrument_id, direction, quantity, entry_price, entry_date, exit_price, exit_date, pnl |
| `PaperPortfolio` | id, name, initial_cash, currency, strategy_config, started_at |
| `PaperHolding` | paper_portfolio_id, instrument_id, quantity, average_cost |
| `PaperOrder` | paper_portfolio_id, instrument_id, direction, quantity, simulated_fill_price, placed_at |

### Alerts & Monitoring

| Entity | Key Attributes |
|---|---|
| `AlertDefinition` | id, name, alert_type, condition_config (JSON), is_active |
| `AlertEvent` | alert_id, instrument_id, triggered_at, message, resolved_at |
| `AuditLog` | id, user_id, action, entity_type, entity_id, before_state, after_state, timestamp |

---

## 8. API BOUNDARY MAP

| Route Group | Responsible For | Notable Endpoints |
|---|---|---|
| `/api/v1/market` | OHLCV bars, tickers, market status, market calendar, exchanges | `GET /bars/{symbol}`, `GET /calendar/{exchange}` |
| `/api/v1/stocks` | Instrument search, security master, company profiles, corporate actions | `GET /search`, `GET /{symbol}/profile` |
| `/api/v1/fundamentals` | Financial statements, ratios, metrics (PIT-aware) | `GET /{symbol}/income`, `GET /{symbol}/ratios` |
| `/api/v1/features` | Feature values, feature definitions, feature history | `GET /{symbol}/features`, `GET /definitions` |
| `/api/v1/predictions` | Latest predictions, prediction history, prediction strength | `GET /{symbol}/latest`, `GET /{symbol}/history` |
| `/api/v1/research` | LLM explanations, factor breakdowns, comparison, Q&A | `POST /explain`, `GET /compare` |
| `/api/v1/portfolio` | Portfolio state, holdings, allocation, optimization | `GET /{id}/holdings`, `POST /{id}/optimize` |
| `/api/v1/risk` | Portfolio risk metrics, instrument risk, stress tests | `GET /{id}/metrics`, `POST /stress-test` |
| `/api/v1/signals` | Active signals, signal history, lifecycle transitions | `GET /active`, `GET /{id}` |
| `/api/v1/backtests` | Run backtests, retrieve results, compare runs | `POST /run`, `GET /{id}/results` |
| `/api/v1/paper-trading` | Paper portfolio, paper orders, P&L | `POST /order`, `GET /{id}/pnl` |
| `/api/v1/alerts` | Alert definitions, alert events, acknowledge | `POST /definitions`, `GET /events` |

> All routes: authenticated, rate-limited, schema-validated, audit-logged.
> All read endpoints: PIT-aware where applicable.

---

## 9. DATA PIPELINE

```
PROVIDER CALL
│
├─ Raw response received
│   └─ ProvenanceRecord created:
│       event_time / publication_time / retrieval_time / availability_time
│
├─ SOURCE VERIFICATION
│   ├─ Is the provider known and trusted?
│   ├─ Is the response schema correct?
│   └─ Does the data match the requested instrument/period?
│
├─ DATA QUALITY ENGINE
│   ├─ Schema check
│   ├─ Type check
│   ├─ Missing values check
│   ├─ Duplicate check
│   ├─ Timestamp consistency check
│   ├─ Range check
│   ├─ Staleness check
│   ├─ Outlier detection
│   └─ Cross-source conflict check (if multiple providers)
│       └─ DataQualityRecord stored → Quality State resolved
│
├─ NORMALIZATION
│   ├─ Currency normalization
│   ├─ Timezone normalization (all stored in UTC)
│   ├─ Adjusted prices (split/dividend adjustment)
│   └─ Units normalization (billions, percentages, ratios)
│
└─ STORAGE
    ├─ Written to appropriate table with quality_state + provenance
    └─ Event published to event bus: DataUpdatedEvent
```

---

## 10. ML PIPELINE

```
FEATURE STORE (PIT-safe)
│
├─ DATASET BUILDER
│   ├─ Select instruments (universe)
│   ├─ Select features (with version)
│   ├─ Select date range
│   ├─ Apply PIT filtering (no future data)
│   └─ Construct target variable (forward return / direction)
│
├─ TRAIN / VAL / TEST SPLIT
│   ├─ Walk-forward
│   ├─ Expanding window
│   └─ Rolling window
│
├─ MODEL TRAINING
│   ├─ Hyperparameter search
│   ├─ Cross-validation (time-series aware)
│   └─ Model artifact saved + registered
│
├─ MODEL EVALUATION
│   ├─ Out-of-sample metrics (Sharpe, IC, hit rate, calibration)
│   ├─ Regime-conditional performance
│   ├─ Overfitting diagnostics
│   └─ Comparison with baseline models
│
├─ EXPERIMENT RECORD
│   ├─ dataset_id
│   ├─ feature_version
│   ├─ model_version_id
│   ├─ hyperparameters
│   ├─ training/val/test periods
│   ├─ seed
│   ├─ metrics
│   └─ git_commit
│
└─ MODEL REGISTRY PROMOTION
    DEVELOPMENT → VALIDATION → CANDIDATE → PRODUCTION → RETIRED
```

---

## 11. PORTFOLIO / DECISION PIPELINE

```
PREDICTION ENGINE
│ expected_return, direction, prediction_strength, uncertainty
▼
EXPLANATION ENGINE
│ factor contributions, SHAP, model agreement
▼
RISK ENGINE
│ instrument risk: vol, beta, VaR, liquidity
│ portfolio risk: correlation, concentration, factor exposure
▼
PORTFOLIO ENGINE
│ current holdings + cash + constraints
│ optimizer: equal weight / min variance / max Sharpe / HRP / BL
│ position sizing, diversification impact
▼
DECISION ENGINE
│ Considers: prediction + strength + risk + portfolio context +
│            macro regime + data quality + investment horizon +
│            constraints + thesis invalidation conditions
│
├─ BUY     → New signal created (NEW → ACTIVE)
├─ HOLD    → Existing signal maintained
├─ REDUCE  → Signal weakened (ACTIVE → WEAKENED)
├─ EXIT    → Signal closed (ACTIVE/WEAKENED → CLOSED)
└─ WAIT    → No signal, conditions not met
```

---

## 12. REAL-TIME PIPELINE

```
MARKET EVENT (price tick / news / volume spike / earnings)
│
├─ State Manager: update in-memory market state
│
├─ Inference Gate: should we run inference now?
│   ├─ Is this a major price movement? (> N sigma)
│   ├─ Is this important news?
│   ├─ Is there a volume anomaly?
│   ├─ Was a scheduled event triggered (e.g., earnings)?
│   ├─ Has a prediction confidence expired?
│   └─ Has a risk breach occurred?
│
├─ YES → FEATURE UPDATE for affected instruments
│         → PREDICTION ENGINE (selected models, not all)
│         → RISK UPDATE
│         → PORTFOLIO IMPACT CHECK
│         → DECISION ENGINE
│         → Signal lifecycle update (if changed)
│         → Event published: PredictionUpdatedEvent
│
├─ NO → State updated but no inference
│
└─ ALERTS evaluated on every state update
    → AlertEvent created if triggered
    → Notification queued for frontend/user
```

---

## 13. MODEL LIFECYCLE

```
DEVELOPMENT
  ─ Experimenting with features, hyperparameters
  ─ No production access
  ─ All experiments tracked in experiment registry

VALIDATION
  ─ Formal OOS evaluation
  ─ Regime-conditional testing
  ─ Bias checks passed
  ─ Calibration verified

CANDIDATE
  ─ Shadow-running alongside production model
  ─ Paper-trading performance tracked
  ─ Model monitor active

PRODUCTION
  ─ Generating live signals
  ─ Fully monitored
  ─ Drift alerts active
  ─ Rollback plan ready

RETIRED
  ─ Superseded by better model
  ─ Artifacts preserved for audit
  ─ Historical predictions retained
```

Every stage transition requires:
- Metric threshold
- Human review (Phase 12+: optional automation)
- Git commit recorded
- Experiment ID linked

---

## 14. DATA LINEAGE

Every prediction must be traceable to:

```
Prediction
  └─ ModelVersion
      ├─ git_commit
      ├─ training_period
      ├─ feature_version → FeatureDefinition (formula version)
      └─ dataset_id
          └─ instruments + date_range + split_config
              └─ Raw data sources
                  └─ ProvenanceRecord (event_time, publication_time, availability_time)
                      └─ DataQualityRecord (what checks passed/failed)
```

This chain ensures:
- Full reproducibility of any historical prediction
- Auditability for any investment decision
- Debugging capability for any model failure

---

## 15. TESTING ARCHITECTURE

| Test Type | Location | What It Tests | Key Rule |
|---|---|---|---|
| **Unit — Domain** | `tests/unit/domain/` | Entity construction, value object equality, enum coverage | No DB, no HTTP |
| **Unit — Services** | `tests/unit/services/` | Factor math, feature transforms, risk formulas | Mocked repositories |
| **Unit — Models** | `tests/unit/models/` | Model fitting on tiny synthetic data | No real data |
| **Integration** | `tests/integration/` | DB persistence, pipeline stages together | Test DB (Postgres) |
| **Financial Correctness** | `tests/financial/` | PIT integrity, backtest bias, factor math accuracy | Domain expert review required |
| **Provider Contract** | `tests/contracts/` | Provider returns expected schema/types | Run against real API in CI if available |
| **API** | `tests/integration/test_api_routes.py` | HTTP status, schema validation, auth | TestClient (no real DB needed for unit) |
| **End-to-End** | `tests/e2e/` | Full pipeline: ingest → predict → signal | Real test environment |
| **Regression** | Part of CI | Key metrics don't regress after code change | Tracked in experiment registry |

> **Financial correctness tests are architecturally separate from software tests.**
> They verify the math is right, not that the code runs. They require domain-level review.

---

## 16. GIT / GITHUB / DOCKER STRUCTURE

### Git
- **Branch strategy**: `main` (production) → `develop` → `feature/*`, `fix/*`, `experiment/*`
- **Commits linked to experiments**: every model training run records `git_commit` hash
- **Tags**: mark model promotions to PRODUCTION stage
- **No git/ folder needed** — Git is a tool, not a source artifact

### GitHub
- **Remote origin**: `github.com/<org>/ai-stock-system`
- **Pull Request workflow**: all changes via PR, code review required
- **`.github/workflows/ci.yml`**: on PR → lint (ruff, black) + type-check (mypy) + unit tests + financial tests
- **`.github/workflows/cd.yml`**: on merge to main → build Docker image + deploy
- **No github/ folder needed** — GitHub config lives in `.github/`

### Docker
- **`docker/docker-compose.yml`**: `backend` + `postgres` + `redis` + `worker` for local development
- **`backend/Dockerfile`**: backend image (Python, dependencies, app)
- **Separate services**:
  - `backend`: FastAPI app server (uvicorn)
  - `worker`: Celery/async task worker (ingestion, heavy computation)
  - `postgres`: PostgreSQL (or Supabase in prod)
  - `redis`: Cache + event bus + task queue
- **No app code in docker/** — only orchestration YAML and config

---

## 17. PHASE 1–16 ROADMAP

| Phase | Name | What You Implement | Depends On |
|---|---|---|---|
| **1** | Foundation | `core/`, `domain/entities/`, `domain/enums/`, `domain/value_objects/`, `domain/interfaces/`, `pyproject.toml`, Docker, `.github/`, `README` | Nothing |
| **2** | Database Layer | SQLAlchemy models, Alembic setup + initial migration, repositories, session factory, Supabase connection | Phase 1 |
| **3** | Data Ingestion | Provider interfaces implementation (dev: yfinance), data verification, data quality engine, provenance tracking, ingestion use-cases | Phase 2 |
| **4** | Factor & Feature Engine | All factor calculators, feature engineering transforms, feature store service, feature versioning | Phase 3 |
| **5** | ML Model Foundation | Dataset builder, PIT-safe splitter, classical models, LSTM/GRU/Transformer, FinBERT, experiment tracker | Phase 4 |
| **6** | Prediction & Explainability | Prediction engine, uncertainty estimator, calibrator, prediction strength calculator, SHAP explainer, LLM narrator | Phase 5 |
| **7** | Risk & Portfolio Engines | Risk engine (all metrics), portfolio optimizers, position sizer, constraints checker | Phase 6 |
| **8** | Decision Engine | Decision engine, signal manager, thesis invalidator, signal lifecycle | Phase 7 |
| **9** | Backtesting | Event-driven backtester, bias guard, performance calculator, walk-forward validation | Phase 5–8 |
| **10** | Paper Trading | Paper portfolio engine, paper order management, P&L tracker, prediction accuracy tracking | Phase 8–9 |
| **11** | Real-Time & Alerts | Event bus, real-time event processor, inference gate, state manager, all alert evaluators | Phase 8–10 |
| **12** | Model Monitoring | Model monitor, data drift monitor, system health monitor | Phase 11 |
| **13** | API Layer | FastAPI app, all routes, schemas, auth, middleware, rate limiting | Phase 2–11 |
| **14** | Security & Auth | JWT, RBAC, secret management, audit logging completion | Phase 13 |
| **15** | Frontend | Next.js app (dashboard, predictions, portfolio, signals, research) | Phase 13 |
| **16** | Production Hardening | Production provider adapters, CI/CD completion, monitoring dashboards, performance tuning | All |

---

## 18. DECISIONS REQUIRING YOUR APPROVAL

Before Phase 1 implementation begins, please confirm:

1. **Production data provider**: Which provider(s) for NSE/BSE market data, fundamentals, news? (Evaluate after Phase 3 with yfinance dev provider)
2. **Real-time vs batch-first**: Should Phase 3 target real-time ingestion, or batch-first with real-time added in Phase 11?
3. **Supabase vs self-hosted Postgres**: Supabase for database management (connection pooling via Supabase), or plain Docker Postgres?
4. **Task queue**: Celery + Redis, or native async (FastAPI background tasks), or a managed queue (e.g., Dramatiq)?
5. **Experiment tracker**: Custom (Phase 5) vs MLflow integration?
6. **LLM for narration**: OpenAI API, local model (Ollama), or optional/deferred?
7. **Redis vs in-process event bus**: For development, in-process is simpler. Confirm Redis for Phase 11.
8. **Universe definition**: NSE only? NSE + BSE? Nifty 500 universe? Global coverage?
9. **Investment horizons to implement first**: Day 1, Day 5, Month 1? All horizons from start?
10. **Authentication scope**: Single-user local system, or multi-user with role-based access?

---

## 19. THINGS INTENTIONALLY NOT DECIDED YET

The following are deliberately deferred:

- Which production market data provider to use
- Exact factor selection (all factors listed are candidates, not commitments)
- Which ML models will actually be promoted to production (requires experimentation)
- Exact hyperparameter search strategy (grid, random, Optuna, etc.)
- Specific portfolio optimization method(s) to ship first
- Whether Black-Litterman will be implemented (depends on data availability)
- Alternative data sources (depends on legality, reliability, cost evaluation)
- Intraday trading support (significant additional complexity)
- Multi-currency support scope
- Frontend design system details (handled when Phase 15 begins)
- Push notification delivery mechanism (WebSocket, email, SMS)
- Whether to integrate a monitoring platform (Prometheus/Grafana) or build custom

---

## 20. PHASE 0 DEFINITION OF DONE

Phase 0 is complete when:

- [x] Architecture overview is understood and agreed upon
- [x] All 16 subsystems have clear boundaries defined
- [x] Repository skeleton is approved (folder tree + responsibilities)
- [x] Domain entity map is accepted
- [x] API contract groups are accepted
- [x] Data pipeline is clearly defined
- [x] ML pipeline is clearly defined
- [x] Point-in-time strategy is understood
- [x] Data quality states are agreed upon
- [x] Prediction Strength concept (not arbitrary %) is agreed upon
- [x] LLM role is understood (explainer only, not predictor)
- [x] Decision engine philosophy is understood (not prediction → auto-buy)
- [x] Phase 1–16 roadmap is reviewed
- [x] Open decisions (Section 18) are answered by you
- [x] All intentionally deferred decisions (Section 19) are acknowledged
- [ ] **You give approval to proceed to Phase 1**

---

*Document version: Phase 0.1 — awaiting review and approval.*
