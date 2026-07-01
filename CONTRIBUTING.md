# Contributing to mytheus-intelligence

## Naming comes from canon — never invent
Service names, the six agent class names, Qdrant collection names, the AGE graph `nexus_knowledge`
and its edge labels, and topic names are frozen in `mytheus-docs/_canon/` and published in
`24-glossary-and-naming.md`. Use them verbatim. A rename goes **Glossary + ADR first** in
`mytheus-docs`, then propagates here — never the reverse.

**Canon location (recorded decision):** canon authority lives in `mytheus-docs`. For MVP the
contract-binding repos consume it as a **sibling checkout** — clone all Mytheus repos under one
parent directory. Revisit git-submodule / vendoring post-W36.

## Isolation and confirmation invariants
This tier NEVER writes Postgres directly and NEVER produces `decisions`/`actions` — it emits
`agent-results`; `core-api` commits on user confirm. `org_id` rides in gRPC metadata on every mesh
hop and in every event header; agent-to-agent authz is deny-by-default, same-tenant-only. Consume
Avro contracts from `mytheus-events`; never hand-roll serializers.

## Every PR includes an `org_id` authz review
Confirm new retrieval / graph / embedding paths filter by `org_id` (per-tenant Qdrant collections,
`$org_id`-parameterized Cypher, FTS RLS) and that no path writes Postgres or the graph without a
confirmed event. Shift-left security from day one.

## Standards
- **Conventional Commits**; **no `Co-Authored-By` footer** (per this repo's CLAUDE.md).
- **Trunk-based**: short-lived branches, frequent PRs, mandatory review.
- CI green before merge: `make ci` (ruff + black + mypy + pytest). Install hooks once:
  `uv run pre-commit install`.
