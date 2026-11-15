# A2A and agent-layer red team

The harness behind the isolation pillar's agent-layer section: insider exfiltration through
agent-to-agent (A2A) context recovery, query steering, indirect and spoken prompt injection,
ProactiveAgent alert suppression, bus-level forgery and budget denial of service. It runs the
real `ContextGapAgent`, PEP, signed bus messages, `Responder`, `DecisionAgent`, `ActionAgent`
and the ProactiveAgent conflict prompt; only retrieval, the PDP, the broker and the model
transport are stand-ins. Results, the corpus (canaries included) and cached model responses
live outside this repository, in the journal's metrics folder (`research/metrics/redteam/`).

| File | What it is |
| --- | --- |
| `org.py` | A synthetic company (12 people, 6 teams, 37 sources) with the ground-truth oracle `may_learn(user, source)` and canary facts in confidential sources |
| `rag.py` | Participant-scoped lexical retrieval over the org (ACL enforced, scores on 0..1) |
| `corpus.py` | About 30 hand-written seeds and the attack styles (AgentDojo, InjecAgent, ConfusedPilot, fake delimiter) over five carriers; benign gaps for utility |
| `paraphrase.py` | Haiku paraphrases of the exfiltration seeds, kept only if they still trip the detector |
| `llm.py` | Record-and-batch model backend: one Message Batch per pass, every answer stored, spend in the shared ledger |
| `harness.py` | Runs instances under a condition (grant table, class filter, retrieval identity, code flags, prompt pins) |
| `fixes.py` | Post-fix conditions: all controls on, leave-one-out ablations, extractive option |
| `analyze.py` | Oracles (canary leak, payload, paired suppression, benign correctness) and Wilson intervals |
| `adaptive.py`, `prompts/attacker.md` | One adaptive attacker loop (Sonnet rewrites failed attacks knowing the defences) |
| `bandwidth.py` | Exfiltration bandwidth per meeting and T8 gap-flooding cost, with a perfectly leaky scripted model ($0) |
| `protocol_attacks.py` | T7: response-body forgery before/after, eavesdropping, service-identity trust assumption, planted cache entries |
| `report.py`, `run.py` | Tables, figures and the driver |

## Run

```bash
uv run python -m research.redteam.run corpus
uv run python -m research.redteam.run baseline --runs 3   # from the pre-fix commit (B1 = origin/main)
uv run python -m research.redteam.run fixes --runs 3
uv run python -m research.redteam.run adaptive --rounds 3
uv run python -m research.redteam.bandwidth --metrics <metrics>/redteam
uv run python -m research.redteam.protocol_attacks --out <metrics>/redteam/raw
uv run --with matplotlib python -m research.redteam.run report
```

`--metrics` defaults to the journal metrics folder; the API key comes from `--env-file`. Every
batch is checked against the `redteam` stream cap before it is sent.

## Conditions

- B0: every grant, no class-to-kind filter (no PEP).
- B1: origin/main, the tenant adopted the platform grant template at org scope. B1t: the same at
  same_team scope.
- B2: the original design, the responder retrieves as the requester
  (`A2A_RESPONDER_AS_REQUESTER=1`).
- B3: no A2A (`CONTEXT_GAP_A2A=0`).
- FIX: every control on; `FIX-<control>` leaves one out; `FIX+extract` adds the extractive-only
  fill; `B1+spot` is B1 with spotlighting only.

The production flags are documented in `services/agent_runtime_service/a2a/pep.py`,
`a2a/identity.py`, `agents/context_gap_agent.py` and `libs/shared/llm_cache.py`.
