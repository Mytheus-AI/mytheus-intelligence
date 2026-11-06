"""Deterministic, dependency-free detector for structured identifiers.

Covers EMAIL, PHONE, IBAN (mod-97), CARD (Luhn), GOVID (US-SSN shape). It cannot find PERSON /
ORG / LOC; those need the model. Used (a) to supplement the model on every request and (b) alone
when the caller opts into degraded mode while the model is unavailable.
"""

from __future__ import annotations

import re

from services.ner_service.spans import Span

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{2,4}){3,8}\b")
_GOVID = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PHONE = re.compile(
    r"(?<![\w])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,4}(?!\w)"
)


def luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(raw: str) -> bool:
    s = raw.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    try:
        num = "".join(str(int(c, 36)) for c in rearranged)
    except ValueError:
        return False
    return int(num) % 97 == 1


def detect_regex(text: str) -> list[Span]:
    out: list[Span] = []
    for m in _EMAIL.finditer(text):
        out.append(Span("EMAIL", m.start(), m.end()))
    for m in _CARD.finditer(text):
        if luhn_ok(re.sub(r"[ -]", "", m.group())):
            out.append(Span("CARD", m.start(), m.end()))
    for m in _IBAN.finditer(text):
        if iban_ok(m.group()):
            out.append(Span("IBAN", m.start(), m.end()))
    for m in _GOVID.finditer(text):
        out.append(Span("GOVID", m.start(), m.end()))
    for m in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 7 <= len(digits) <= 15:
            out.append(Span("PHONE", m.start(), m.end(), score=0.8))
    return out
