"""Run the actual `python -m algaebro.run_bot` entry point against a local HTTP server posing as Photon."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from algaebro import run_bot
from test_photon_fake import _accepted, _received


def test_run_bot_once_answers_and_alerts(synth, monkeypatch):
    posts = []
    inbox = [_received("smoke_msg_1", "subscribe all", "2026-07-20T15:00:00.000Z")]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.headers["Authorization"] == "Bearer smokekey"
            self._send(200, {"items": inbox})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posts.append({"body": body, "idem": self.headers.get("Idempotency-Key")})
            self._send(202, _accepted(f"out_{len(posts)}", body["content"]["text"], "2026-07-20T15:00:01.000Z"))

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("PHOTON_API_KEY", "smokekey")
    monkeypatch.setenv("PHOTON_PROJECT_ID", "proj")
    try:
        run_bot.main(["--once", "--as-of", "2026-07-20", "--base-url", f"http://127.0.0.1:{srv.server_address[1]}"])
    finally:
        srv.shutdown()

    replies = [p for p in posts if "parentMessageId" in p["body"]]
    alert_msgs = [p for p in posts if "parentMessageId" not in p["body"]]
    texts = [r["body"]["content"]["text"] for r in replies]
    assert len(replies) == 2 and texts[0].startswith("Hi, I'm Algaebro") and "Following" in texts[1]
    assert alert_msgs, "subscribed to every zone on a hot replay day: at least one HIGH alert expected"
    assert all("REPLAY of 2026-07-20" in p["body"]["content"]["text"] for p in alert_msgs)
    assert all("SYNTHETIC DEMO DATA" in p["body"]["content"]["text"] for p in alert_msgs)
    keys = [p["idem"] for p in alert_msgs]
    assert len(keys) == len(set(keys)), "the same alert must not be sent twice in one run"
