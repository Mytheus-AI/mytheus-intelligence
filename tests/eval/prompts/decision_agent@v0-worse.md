DELIBERATELY WORSE BASELINE (eval fixture, never shipped). It exists so the offline eval can show it scores below the current prompt: it asks for recall over precision, counts proposals and tasks as decisions, gives no calibration guidance and ignores the company ground rules.

List every decision, plan, idea or option mentioned in 30 seconds of a meeting transcript. Include anything anyone suggested, since the user can dismiss it. Speakers appear as "Participant N".

Answer with JSON: {"candidates": [{"content": "...", "rationale": null, "stakeholders": [], "confidence": 0.0}]}
Use a high confidence for anything that was said clearly.
---user---
Transcript (30 seconds):
{window}
