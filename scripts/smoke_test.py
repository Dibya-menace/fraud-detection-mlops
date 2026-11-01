"""Smoke test: call a running service. Usage: python scripts/smoke_test.py <base_url>"""
import json
import sys
import urllib.request

base = sys.argv[1].rstrip("/")

feats = {f"V{i}": 0.0 for i in range(1, 29)}
feats["Amount"] = 50.0
req = urllib.request.Request(
    f"{base}/predict",
    data=json.dumps({"features": feats}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as r:
    body = json.loads(r.read())
    assert r.status == 200, r.status
print("OK", body)