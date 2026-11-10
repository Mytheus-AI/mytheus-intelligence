DELIBERATELY WORSE BASELINE (eval fixture, never shipped). It treats any two decisions on the same subject as conflicting, including a decision that knowingly replaces an earlier one, and gives no calibration guidance.

Decide whether the new decision conflicts with any earlier decision. Decisions about the same subject usually conflict.

Answer with JSON: {"conflicts": [{"candidate": 1, "confidence": 0.0, "rationale": "..."}]}
---user---
New decision:
{decision}

Earlier decisions:
{candidates}
