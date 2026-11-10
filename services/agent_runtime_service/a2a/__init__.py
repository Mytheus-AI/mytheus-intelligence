"""Agent-to-agent collaboration for `ContextGapAgent` (mytheus-docs ADR-0013, plan 10-13 3.3).

- `identity`: the bound agent identity `cga:{org_id}:{user_id}:{sig}`, minted and verified only
  here (HMAC-SHA256 over `AGENT_IDENTITY_SECRET`).
- `pep`: the policy enforcement point. Deterministic `query_class` / `target_scope`, the 60 s
  decision cache at `nexus:{org_id}:a2a:grant:...`, and the PDP over identity gRPC `Authorize`.
- `bus`: `agent-requests` / `agent-responses` (requester client with in-process futures, the
  responder with its pre-process checks, and the two consumer loops).
- `collab`: the `agent_collaborations` entry shape both sides emit on `agent-results`.
- `related`: core-api's related-attendees route for the related-meeting hop.
"""
