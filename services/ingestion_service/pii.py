"""Re-export. The redactor moved to `libs/shared/pii.py` so `agent-runtime-service` can apply
the same gate before LLM egress (Plan 7-9 section 3.3 step 4) without importing another
service's package."""

from libs.shared.pii import pii_enabled, redact_pii

__all__ = ["pii_enabled", "redact_pii"]
