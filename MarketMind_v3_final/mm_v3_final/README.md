# MarketMind

A sophisticated multi-layer AI trading signal system that combines swarm simulation with structured agent-based analysis to generate auditable, forward-looking trading signals for Indian equities and futures.

---

## The Core Idea

Traditional quant models analyse price and fundamentals. MarketMind adds a third dimension: **simulated crowd psychology**. A population of thousands of AI agents — each with distinct personas, risk tolerances, and behavioral biases — reacts to the same market event you are evaluating. The emergent output of that simulation becomes one of the inputs to a structured trading desk of specialist agents, which synthesises everything into a single, typed, auditable trading signal.

The signal is a recommendation for you to review. You decide whether to act. The system never touches your broker automatically.

---

## Architecture

```
Data Inputs (market data, news, macro, sentiment, positioning)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1 — Digital World (Swarm Simulation)         │
│                                                     │
│  Tier 3: 10,000+ Mesa ABM agents  (rule-based, ~$0) │
│  Tier 2: 200 GPT-4o-mini agents   (async batch)     │
│  Tier 1: 10 GPT-4o agents         (deep reasoning)  │
│                                                     │
│  Output: SimulationReport (emergent crowd dynamics) │
└─────────────────────────────────────────────────────┘
         │
         │  SimulationReport
         ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 2 — Trading Desk (LangGraph)                 │
│                                                     │
│  ├─ Macro Strategist     (GPT-4o, parallel)         │
│  ├─ Technical Analyst    (GPT-4o, parallel)         │
│  ├─ Risk Manager         (GPT-4o, parallel)         │
│  └─ Sentiment Interpreter (GPT-4o, reads sim report)│
│                                                     │
│  └─ Desk Coordinator     (Claude — model diversity) │
│                                                     │
│  Output: TradingSignal (Pydantic, fully typed)      │
└─────────────────────────────────────────────────────┘
         │
         ▼
  You review → You decide → Optional: broker API
```

---

## Project Structure

```
market_mind/
│
├── models/
│   ├── signal.py          ← TradingSignal — the final output contract
│   └── simulation.py      ← SimulationReport — Layer 1 → Layer 2 handoff
│
├── simulation/            ← Layer 1: Digital World
│   ├── tiers/
│   │   ├── tier3_mesa.py  ← 10,000+ rule-based ABM agents (Mesa 3)
│   │   ├── tier2_async.py ← 200 GPT-4o-mini agents (asyncio batch)
│   │   └── tier1_deep.py  ← 10 GPT-4o deep thinker agents
│   ├── personas.py        ← Agent persona definitions
│   ├── coordinator.py     ← Aggregates tier outputs → SimulationReport
│   └── runner.py          ← Entry point for simulation runs
│
├── desk/                  ← Layer 2: Trading Desk (LangGraph)
│   ├── agents/
│   │   ├── macro.py       ← Macro Strategist agent
│   │   ├── technical.py   ← Technical Analyst agent
│   │   ├── risk.py        ← Risk Manager agent
│   │   └── sentiment.py   ← Sentiment Interpreter agent
│   ├── coordinator.py     ← Head Trader (Claude) — final synthesis
│   └── graph.py           ← LangGraph state machine definition
│
├── pipeline/
│   ├── runner.py          ← End-to-end pipeline orchestration
│   ├── data_ingestion.py  ← Market data, news, positioning feeds
│   └── signal_logger.py   ← Logs signals to CSV + JSONL
│
├── utils/
│   ├── ids.py             ← ULID-based signal and run ID generation
│   └── logging.py         ← Structured logging setup
│
├── tests/
│   ├── test_models.py     ← Schema validation tests
│   ├── test_simulation.py ← Simulation unit tests
│   └── test_desk.py       ← Desk agent unit tests
│
├── config.py              ← All configuration in one place
├── requirements.txt       ← Pinned dependencies
├── .env.example           ← Environment variable template
└── README.md              ← This file
```

---

## Build Order

We build in this order deliberately. Each step is independently testable before the next begins.

| Step | What | Why first |
|------|------|-----------|
| ✅ 1 | Signal + Simulation schemas (Pydantic) | Defines what we're building toward |
| ✅ 2 | `config.py` | All constants in one place before any logic |
| 🔲 3 | Tier 3 Mesa ABM | Cheapest, fastest, no API keys needed — validates crowd model |
| 🔲 4 | Signal logger + IDs | Can log Tier 3 output immediately |
| 🔲 5 | Tier 2 async batch | Add informed crowd on top of Mesa crowd state |
| 🔲 6 | Tier 1 deep thinkers | Add the sophisticated actors |
| 🔲 7 | Simulation coordinator | Aggregates tiers → SimulationReport |
| 🔲 8 | Desk agents (parallel) | Macro, Technical, Risk, Sentiment |
| 🔲 9 | Desk coordinator (Claude) | Final synthesis → TradingSignal |
| 🔲 10 | LangGraph graph | Wire desk into a proper state machine |
| 🔲 11 | Pipeline runner | End-to-end orchestration |
| 🔲 12 | Dashboard | Review UI for signals |

---

## The One Hard Rule

**This system does not backtest. Ever.**

Every agent is given today's date as `point_in_time_date` and instructed to reason only with information available as of that date. LLMs have memorized historical market data within their training window — a backtest would be contaminated by recall, not reasoning. We validate this system exclusively through forward paper trading over a minimum 3–6 month window.

The `TradingSignal.point_in_time_date` and `SimulationReport.point_in_time_date` fields are validated against `date.today()` at creation time and will raise immediately if you try to pass a historical date.

---

## Signal Schema Quick Reference

```python
TradingSignal(
    instrument      = "NIFTY50",
    direction       = Direction.LONG,
    conviction      = 0.78,          # 0–1, actionable >= 0.70
    timeframe       = Timeframe.SWING,
    entry_zone      = PriceZone(low=24200, high=24350),
    risk            = RiskParameters(
        stop_loss        = 23900,
        target_1         = 24900,
        risk_reward      = 2.3,
        max_position_pct = 0.04,      # 4% of portfolio
        invalidation_level = 23700,
    ),
    crowd           = CrowdMetrics(
        aggregate_sentiment  = -0.35,  # fear in the crowd
        herd_index           = 0.71,   # high herding
        panic_probability    = 0.42,
        narrative_momentum   = 0.60,   # narrative spreading fast
        contrarian_pressure  = 0.25,   # 25% of deep thinkers disagree
        ...
    ),
    do_not_trade_flags = [],           # empty = cleared for review
    ...
)

signal.is_actionable    # True if conviction >= 0.70 and no vetoes
signal.signal_summary   # one-line readable summary
signal.crowd_summary    # one-line crowd intelligence summary
```

---

## Cost Estimate Per Signal

| Layer | Cost |
|-------|------|
| Tier 3 (10,000 Mesa agents) | ~$0 |
| Tier 2 (200 × GPT-4o-mini) | ~$0.50–1.50 |
| Tier 1 (10 × GPT-4o) | ~$1.00–2.00 |
| Trading desk (5 × GPT-4o + Claude) | ~$1.50–2.50 |
| **Total** | **~$3–6 per signal** |

---

## Environment Setup

```bash
cp .env.example .env
# Fill in your API keys in .env

pip install -r requirements.txt

# Validate config
python -c "from config import cfg; cfg.validate(); cfg.print_summary()"
```

---

## Philosophy

- **Signals, not holy grails.** The system is one input among many. Your judgment is the last gate.
- **Auditable by design.** Every signal carries a full reasoning chain. No black boxes.
- **Forward-only.** If it can't be tested forward, it doesn't belong here.
- **Typed or it doesn't exist.** Unstructured prose between agents is a debugging nightmare. Every handoff is a Pydantic model.
- **Transparent costs.** Every LLM call is logged. You always know what this is costing you.
