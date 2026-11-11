DELIBERATELY WORSE BASELINE (eval fixture, never shipped). It asks for every mention of work, including wishes and finished work, assigns the first speaker by default, gives no calibration guidance and ignores the company ground rules.

List every task or piece of work mentioned in 30 seconds of a meeting transcript. If the owner is unclear, use "Participant 1". Speakers appear as "Participant N".

Answer with JSON: {"candidates": [{"content": "...", "assignee": "Participant 1", "due_date": null, "depends_on": [], "topic": null, "confidence": 0.0}]}
Use a high confidence for anything that was said clearly.
---user---
Today is {today} (UTC).

Transcript (30 seconds):
{window}
