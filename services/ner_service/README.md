# ner-service

Detects PII entity spans (ADR-0020 D4/D5). **Detects only**: it never redacts, stores, or echoes
text. Callers (transcription, ingestion, agent-runtime/rag) own redaction and token minting.

## Endpoints (all but `/healthz` need `Authorization: Bearer <Clerk JWT>`; org from `libs/shared/auth`)

- `GET /healthz` - liveness, always 200.
- `GET /readyz` - 200 only when the model is loaded (lazy load on first probe/request, retried every 30s); else 503.
- `POST /ner/detect`

Note: canon's catalog lists `POST /ner`; this implements `/ner/detect` per the build task. Align canon or add an alias.

### Request
```json
{"text": "Mail Zoë Müller at zoe@acme.io", "allow_regex_fallback": false}
```
### Response 200
```json
{"spans": [{"type": "PERSON", "start": 5, "end": 14, "score": 0.85, "source": "model"}],
 "model": "spacy:en_core_web_sm", "mode": "model", "partial": false, "offset_unit": "codepoint"}
```
- `type`: `PERSON ORG LOC` (model) and `EMAIL PHONE IBAN CARD GOVID` (deterministic: Luhn / mod-97 validated).
- `start`/`end`: Python code-point offsets, `text[start:end]`. Not UTF-8 bytes or UTF-16 units.
- Spans never overlap; overlapping same-type spans are unioned; across types, validated identifiers beat model output.
- Matched text is never returned.

### Errors
| Status | Meaning |
|---|---|
| 401 | missing/invalid token or no org claim |
| 413 | `text` > `NER_MAX_CHARS` (default 100000) or oversized body |
| 422 | malformed body |
| 503 | **model unavailable (or IdP down). Callers must fail closed** (ADR-0020). |

Degraded mode (ADR-0020 option 2, a compliance decision): send `allow_regex_fallback: true` to get
200 with `mode: "regex_only"`, `partial: true` when the model is down. `partial: true` means
PERSON/ORG/LOC were NOT checked; callers must record the row as partially redacted.

## Config
`NER_MAX_CHARS`, `NER_SPACY_MODEL` (default `en_core_web_sm`), `CLERK_JWKS_URL`, `CLERK_ISSUER`,
`CLERK_JWT_INTERNAL_ORG_CLAIM`. Install the model with the `ner` extra
(`uv sync --extra ner && python -m spacy download en_core_web_sm`); without it the service runs
but is not ready. Docker: `docker build -f services/ner_service/Dockerfile .` (CPU, non-root).
Logs carry org_id, lengths, counts only, never text.
