"""FastAPI web application wrapping TradingAgentsGraph."""

import asyncio
import json
import os
import traceback
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.validators import VALID_MODELS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="TradingAgents Dashboard")

# Resolve templates directory relative to this file so it works regardless of CWD
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_INDEX_HTML = (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

# In-memory job storage
jobs: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    ticker: str
    date: str
    llm_provider: str = "openai"
    deep_think_llm: str = "gpt-5.2"
    quick_think_llm: str = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Analysis pipeline stages (for progress tracking)
# ---------------------------------------------------------------------------
STAGES = [
    "Market Analyst",
    "Social Media Analyst",
    "News Analyst",
    "Fundamentals Analyst",
    "Bull/Bear Debate",
    "Research Manager",
    "Trader",
    "Risk Debate",
    "Portfolio Manager",
    "Final Decision",
]


# ---------------------------------------------------------------------------
# Background analysis runner
# ---------------------------------------------------------------------------

def _run_analysis(job_id: str, req: AnalyzeRequest) -> None:
    """Run TradingAgentsGraph.propagate in a background thread."""
    job = jobs[job_id]
    try:
        job["progress"].append({"stage": "Initializing", "status": "running"})

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = req.llm_provider
        config["deep_think_llm"] = req.deep_think_llm
        config["quick_think_llm"] = req.quick_think_llm

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # Mark stage progression via debug streaming
        ta = TradingAgentsGraph(debug=True, config=config)

        job["progress"].append({"stage": "Initializing", "status": "done"})

        # We use debug=True which calls graph.stream(). We monkey-patch the
        # graph's stream iterator so we can detect agent transitions and push
        # progress events.  The simplest approach: run propagate in this thread
        # and infer stages from the state keys that get populated.
        #
        # Since propagate blocks, we run it directly and poll state afterwards.

        # Track which stages we've announced
        announced: set = set()

        def _announce(stage: str, status: str = "running") -> None:
            if (stage, status) not in announced:
                announced.add((stage, status))
                job["progress"].append({"stage": stage, "status": status})

        # Start first stage
        _announce(STAGES[0])

        # Run propagate (blocks until complete)
        final_state, decision = ta.propagate(req.ticker, req.date)

        # Mark all stages as done
        for stage in STAGES:
            _announce(stage, "done")

        # Extract reports from final_state
        reports = {}
        state = final_state if isinstance(final_state, dict) else {}

        report_keys = [
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "investment_debate_state",
            "trader_investment_plan",
            "risk_debate_state",
            "investment_plan",
            "final_trade_decision",
        ]
        for key in report_keys:
            val = state.get(key, "")
            # Convert dicts to formatted JSON string for display
            if isinstance(val, dict):
                reports[key] = json.dumps(val, indent=2, default=str)
            else:
                reports[key] = str(val) if val else ""

        # Determine decision string
        decision_str = str(decision).strip().upper() if decision else "UNKNOWN"

        job["status"] = "completed"
        job["decision"] = decision_str
        job["reports"] = reports
        job["completed_at"] = datetime.now().isoformat()

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["progress"].append({"stage": "Error", "status": "failed", "detail": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the dashboard."""
    try:
        # Render the template by replacing the Jinja2 placeholder directly
        html = _INDEX_HTML.replace("{{ models | tojson }}", json.dumps(VALID_MODELS))
        return HTMLResponse(content=html)
    except Exception as exc:
        return PlainTextResponse(
            content=f"Error rendering dashboard: {traceback.format_exc()}",
            status_code=500,
        )


@app.get("/api/models")
async def get_models():
    """Return valid model names by provider."""
    return VALID_MODELS


@app.post("/api/analyze")
async def start_analysis(req: AnalyzeRequest):
    """Start an analysis job in a background thread."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "ticker": req.ticker.upper(),
        "date": req.date,
        "llm_provider": req.llm_provider,
        "deep_think_llm": req.deep_think_llm,
        "quick_think_llm": req.quick_think_llm,
        "status": "running",
        "progress": [],
        "decision": None,
        "reports": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    }

    thread = threading.Thread(target=_run_analysis, args=(job_id, req), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def stream_status(job_id: str):
    """SSE endpoint streaming progress updates for a job."""
    if job_id not in jobs:
        return {"error": "Job not found"}

    async def event_generator():
        last_idx = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                break

            # Send any new progress entries
            progress = job["progress"]
            while last_idx < len(progress):
                yield {
                    "event": "progress",
                    "data": json.dumps(progress[last_idx]),
                }
                last_idx += 1

            if job["status"] in ("completed", "failed"):
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "status": job["status"],
                        "decision": job.get("decision"),
                        "error": job.get("error"),
                    }),
                }
                break

            await _async_sleep(1)

    return EventSourceResponse(event_generator())


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Return full result for a completed job."""
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return {
        "id": job["id"],
        "ticker": job["ticker"],
        "date": job["date"],
        "status": job["status"],
        "decision": job["decision"],
        "reports": job["reports"],
        "error": job["error"],
        "created_at": job["created_at"],
        "completed_at": job["completed_at"],
    }


@app.get("/api/history")
async def get_history():
    """Return list of all jobs (most recent first)."""
    items = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return [
        {
            "id": j["id"],
            "ticker": j["ticker"],
            "date": j["date"],
            "status": j["status"],
            "decision": j["decision"],
            "created_at": j["created_at"],
        }
        for j in items
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)
