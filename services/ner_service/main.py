"""ner-service's synchronous FastAPI process: `/healthz` (liveness, `libs.shared.app`), plus
`/readyz` and `POST /ner` (`detect.py`). Owns no data. Detects only; never redacts.
`/readyz` is 200 only once the model is loaded; `/healthz` stays 200 regardless.
"""

from __future__ import annotations

from libs.shared.app import create_app
from services.ner_service.detect import create_ner_app

SERVICE_NAME = "ner-service"

app = create_app(SERVICE_NAME)
app.mount("/", create_ner_app())
