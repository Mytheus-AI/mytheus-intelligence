"""`ProactiveAgent` plumbing (P11, plan 10-13 section 3.6). The agent itself lives in
`agents/proactive_agent.py`.

- `core`: core-api's internal action-context and meeting-participants routes.
- `loop`: the `decisions` / `actions` consumer (group `agent-runtime-proactive`) that runs the
  conflict, workload and dependency checks and emits the alerts on `agent-results`.
"""
