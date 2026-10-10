# AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System

A production-oriented AI platform combining verified market data, quantitative factor analysis, multi-model ML/DL/NLP predictions, and a risk-gated portfolio decision engine for equity markets.

> **Status**: Phase 1 — Project Skeleton  
> The backend compiles and serves a health endpoint. All domain, service, and API layers are scaffolded and ready for implementation.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the complete approved Phase 0 architecture, including system diagrams, domain boundaries, database entity map, API contract map, and the full Phase 1–16 implementation roadmap.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI |
| Database | PostgreSQL (Supabase-hosted) — Phase 2 |
| ORM | SQLAlchemy 2.x + Alembic — Phase 2 |
| ML | NumPy, Pandas, scikit-learn, XGBoost, LightGBM, PyTorch — Phase 5 |
| NLP | HuggingFace, FinBERT — Phase 5 |
| Quant | SciPy, CVXPY — Phase 7 |
| Frontend | Next.js, TypeScript, Tailwind CSS — Phase 15 |
| Container | Docker, Docker Compose |
| CI | GitHub Actions |

---

## Repository Structure

```
ai-stock-system/
├── backend/
│   ├── app/
│   │   ├── main.py          ← FastAPI entrypoint
│   │   ├── core/            ← Config, logging, exceptions, constants
│   │   ├── domain/          ← Entities, enums, interfaces, value objects
│   │   ├── infrastructure/  ← DB, providers, repositories, messaging
│   │   ├── services/        ← Business logic implementations
│   │   ├── application/     ← Use-case orchestrators
│   │   └── api/             ← Routes, schemas, middleware
│   ├── tests/
│   ├── scripts/
│   ├── Dockerfile
│   └── pyproject.toml
├── docker/
│   └── docker-compose.yml
├── docs/
│   └── architecture.md
└── .github/workflows/ci.yml
```

---

## Development Setup

### Prerequisites

- Python 3.12
- Docker Desktop
- Git

### Local (without Docker)

```bash
# Clone
git clone https://github.com/<org>/ai-stock-system.git
cd ai-stock-system/backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies (including dev tools)
pip install -e ".[dev]"

# Copy env template
cp ../.env.example .env

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at: http://localhost:8000  
Interactive docs: http://localhost:8000/docs  
Health check: http://localhost:8000/health

### Docker

```bash
cd docker
docker compose up --build
```

---

## Running Tests

```bash
cd backend

# Unit tests only (Phase 1)
pytest tests/unit/ -v

# Lint
ruff check .

# Format check
ruff format --check .

# Type check
mypy app/
```

---

## Implementation Phases

| Phase | Name | Status |
|---|---|---|
| 0 | Architecture & Planning | ✅ Complete |
| 1 | Foundation & Skeleton | ✅ Complete |
| 2 | Database Layer | ✅ Complete |
| 3 | Data Ingestion | ✅ Complete |
| 4 | Factor & Feature Engine | 🔲 Not started |
| 5 | ML Model Foundation | 🔲 Not started |
| 6 | Prediction & Explainability | 🔲 Not started |
| 7 | Risk & Portfolio Engines | 🔲 Not started |
| 8 | Decision Engine | 🔲 Not started |
| 9 | Backtesting | 🔲 Not started |
| 10 | Paper Trading | 🔲 Not started |
| 11 | Real-Time & Alerts | 🔲 Not started |
| 12 | Model Monitoring | 🔲 Not started |
| 13 | API Layer | 🔲 Not started |
| 14 | Security & Auth | 🔲 Not started |
| 15 | Frontend | 🔲 Not started |
| 16 | Production Hardening | 🔲 Not started |

---

## Important Design Principles

- **LLMs explain quantitative outputs — they do not generate numerical signals.**
- **Prediction ≠ certainty. Signal ≠ automatic trade.**
- **All historical research uses point-in-time data (no look-ahead bias).**
- **Every prediction is traceable to its data source, model version, and feature version.**
- **The system is not coupled to any single data provider, ML model, or database.**
"# StockVue" 
