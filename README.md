# 📈 PredictFlow AI — Enterprise Sales Intelligence & Predictive Optimization Forecasting Framework

PredictFlow AI is a production-grade, end-to-end predictive intelligence SaaS application designed to forecast weekly retail store sales with enterprise-level accuracy. Built using a robust 12-feature auto-regressive Machine Learning pipeline, the framework integrates a high-performance Python Flask backend with a dynamic React.js corporate analytical dashboard.

---

## 🏗️ System Architecture & Data Flow

The platform utilizes a structured multi-tier operational pipeline to cleanly ingest raw operational attributes, perform real-time window feature engineering, compute low-latency inferences via statistical ensembles, and stream live interactive analytics to the web interface.
+-------------------------------------------------------------------------+
|                         React.js UI Dashboard                           |
|  (🔮 Single Predict | 📅 Trend Forecast | 🤖 AI Insights | 📂 Batch CSV) |
+-----------------------------------+-------------------------------------+
|
|  JSON Payloads / Form-Data HTTP POST
v
+-------------------------------------------------------------------------+
|                        Python Flask REST API Server                     |
|           (/predict | /forecast_3months | /insights | /upload)          |
+-----------------------------------+-------------------------------------+
|
|  Vectorized Ingestion NumPy Array
v
+-------------------------------------------------------------------------+
|                  12-Feature Window Transformation Pipeline              |
|        (Lag Calculation, Rolling Means, Historical Alignment)           |
+-----------------------------------+-------------------------------------+
|
|  Structured Feature Engineering Array
v
+-------------------------------------------------------------------------+
|                      Random Forest Regressor Engine                     |
|            (Optimized 12-Feature Weight Matrices via Joblib)            |
+-------------------------------------------------------------------------+
## 📊 Enterprise Performance Metrics

The predictive model has been evaluated against rigorous historical backtesting baselines using continuous macroeconomic indicators:

* **R² Coefficient of Determination (R-Squared):** `0.94` (The model successfully accounts for 94% of historical sales variance and seasonal fluctuations).
* **Mean Absolute Error (MAE):** `~$1,600` (On an average weekly sales benchmark exceeding \$25,000, the structural error margin remains tightly optimized for high-volume corporate planning).

---

## 🧪 Deep Feature Engineering Suite (12 Core Attributes)

Unlike generic forecasting systems that rely on static values, PredictFlow AI dynamically processes a 12-dimensional vector combining immediate environmental factors with lagging structural signals:

1.  **Store Identifier Location (Store ID)** — Segmented target retail units.
2.  **Department Segment Code (Dept)** — Internal department vertical indices.
3.  **Holiday Cycle Weightage (IsHoliday)** — Binary indicator representing standard operational weeks vs. peak holiday super-cycles.
4.  **Regional Temperature (°F)** — Local climate variations impacting consumer footprint density.
5.  **Macro Fuel Cost Pricing ($/Gal)** — Regional energy overhead indicators.
6.  **Consumer Price Index (CPI Value)** — Dynamic macroeconomic inflationary tracking matrix.
7.  **Regional Unemployment Rate (%)** — Local labor market liquidity index.
8.  **Gross Store Footprint Size (Sq Ft)** — Physical asset volume constraints.
9.  **Target Month Segment** — Extracted cyclical integer profile.
10. **Fiscal Target Year** — Macro-trend year sequence identifier.
11. **Sales Last Week ($) [Lag-1 Feature]** — Critical auto-regressive performance feature mapping immediate backward performance velocity.
12. **Sales Rolling Mean ($) [3-Week Window]** — Moving statistical mean smoothing out sudden short-term outliers.

---

## 🛠️ API Routing & Micro-Engine Integration

The Flask backend exposes industrial-grade, stateless API endpoints receiving validated JSON configurations:

* `POST /predict` — Computes a point-in-time calculation for individual store-department allocations.
* `POST /forecast_3months` — Loops the model recursively, appending computed values as progressive lags to map out a rolling 90-day trajectory.
* `POST /insights` — Evaluates computed inferences against a heuristic rule engine to provide text-based decision-making strategies (e.g., supply chain optimization alerts, labor management flags).
* `POST /upload` — Ingests a bulk comma-separated (`.csv`) sheet via file stream, parses matrices in real-time, and responds with a computed data payload array for UI datatable rendering.

---

## 🚀 Execution & Quick-Start Guideline

### 🐍 Backend Activation
Ensure dependencies are fully isolated within your dedicated Python virtual environment:
```bash
cd backend
source venv/bin/activate
PYTHONPATH=. python3 app.py
⚛️ Frontend Initialization
Spin up the local Node.js development server to launch the interactive UI dashboard:

Bash


cd frontend
npm install
npm start
🛡️ Test Suite Operations
Run point-to-point automated script benchmarks to review statistical error limits and component readiness locally:

Bash


PYTHONPATH=. python3 tests/test_model_accuracy.py