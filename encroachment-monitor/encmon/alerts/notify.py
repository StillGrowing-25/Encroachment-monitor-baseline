"""Alert delivery adapters (paper: SMS / email / API to the responsible officer).

Adapters implement `deliver(alert)`; the console backend logs the payload,
the webhook backend POSTs JSON to a configured endpoint (disabled by
default so the pipeline never calls out unexpectedly).
"""
from __future__ import annotations

import json
import urllib.request


class ConsoleNotifier:
    def __init__(self, channel: str = "SMS/email/API"):
        self.channel = channel

    def deliver(self, alert: dict) -> str:
        line = (f"[{self.channel}] alert #{alert.get('id')} "
                f"parcel {alert.get('parcel_id')} severity={alert.get('severity')} "
                f"status={alert.get('status')} at ({alert.get('lat'):.5f}, {alert.get('lon'):.5f})")
        print(line)
        return line


class WebhookNotifier:
    """POST the alert to an external gateway (SMS/e-mail bridge or API)."""

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def deliver(self, alert: dict) -> str:
        payload = json.dumps(alert, default=str).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return f"webhook {self.url} -> {resp.status}"
