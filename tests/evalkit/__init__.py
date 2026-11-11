"""The offline/live eval harness (polish plan, "Evals that can fail").

Modules:
- `corpus`: the fixture formats and loaders (windows, persona pairs, golden sets).
- `generate`: the committed, seeded generator of the corpus and of the simulated recordings
  (`uv run python -m tests.evalkit.generate`).
- `simulate`: the simulated model behaviour per prompt (what an offline recording says when no
  live recording exists yet).
- `models`: the completion functions the router is given: simulate, replay, live (+ record).
- `scoring`: the metric arithmetic.
- `harness`: runs the real agents per suite and returns a report; `thresholds` checks it.
- `judge`: the LLM-as-judge rubric for note quality (live mode only).
"""
