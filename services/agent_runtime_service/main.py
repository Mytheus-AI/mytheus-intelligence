"""agent-runtime-service: the six LangGraph agents + cost-controlled LLM pipeline.

Scaffold only: boots and answers `/healthz`. The `BaseAgent` lifecycle, tiered routing, per-tenant
budgets, and the `transcript-chunks -> agent-results/agent-stream` flow land in the agent band.

Run: uvicorn services.agent_runtime_service.main:app --port 8200
"""

from __future__ import annotations

from libs.shared.app import create_app

app = create_app("agent-runtime-service")
